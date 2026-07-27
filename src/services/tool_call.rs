use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::{RwLock, Mutex, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{info, error};

/// Tool call state for a pending tool call.
#[derive(Debug, Clone)]
pub struct ToolCallState {
    pub tool_use_id: String,
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub message_id: Option<String>,
}

impl ToolCallState {
    pub fn new(tool_use_id: impl Into<String>, session_id: impl Into<String>, message_id: Option<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            session_id: session_id.into(),
            created_at: Utc::now(),
            message_id,
        }
    }
}

/// Manager for tool call states. Designed to be used behind Arc.
#[derive(Debug)]
pub struct ToolCallManager {
    inner: Arc<RwLock<HashMap<String, ToolCallState>>>,
    cleanup_handle: Mutex<Option<JoinHandle<()>>>,
    shutdown_tx: Mutex<Option<watch::Sender<bool>>>,
    timeout: Duration,
    cleanup_interval: Duration,
}

impl ToolCallManager {
    /// Create a new manager with explicit timeout and cleanup interval.
    pub fn new(timeout: Duration, cleanup_interval: Duration) -> Self {
        info!("ToolCallManager initialized with timeout={:?}, cleanup_interval={:?}", timeout, cleanup_interval);
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            cleanup_handle: Mutex::new(None),
            shutdown_tx: Mutex::new(None),
            timeout,
            cleanup_interval,
        }
    }

    /// Register a tool call.
    pub async fn register_tool_call(&self, tool_use_id: String, session_id: String, message_id: Option<String>) {
        let state = ToolCallState::new(tool_use_id.clone(), session_id.clone(), message_id);
        let mut map = self.inner.write().await;
        map.insert(tool_use_id.clone(), state);
        info!("Registered tool call: {} for session: {}", tool_use_id, session_id);
    }

    /// Get a tool call by id.
    pub async fn get_tool_call(&self, tool_use_id: &str) -> Option<ToolCallState> {
        let map = self.inner.read().await;
        map.get(tool_use_id).cloned()
    }

    /// Mark a tool call as complete (removes it).
    pub async fn complete_tool_call(&self, tool_use_id: &str) {
        let mut map = self.inner.write().await;
        if map.remove(tool_use_id).is_some() {
            info!("Completed tool call: {}", tool_use_id);
        } else {
            info!("Attempted to complete unknown tool call: {}", tool_use_id);
        }
    }

    /// Start the background cleanup task (idempotent).
    pub async fn start_cleanup_task(self: &Arc<Self>) {
        let mut handle_guard = self.cleanup_handle.lock().await;
        if handle_guard.is_some() {
            // already running
            return;
        }

        let (tx, mut rx) = watch::channel(false);
        *self.shutdown_tx.lock().await = Some(tx.clone());

        let this = Arc::clone(self);
        let timeout = self.timeout;
        let cleanup_interval = self.cleanup_interval;

        let handle = tokio::spawn(async move {
            info!("Started tool call cleanup task");
            let mut interval = tokio::time::interval_at(Instant::now() + cleanup_interval, cleanup_interval);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = this.cleanup_expired_tool_calls(timeout).await {
                            error!("Error during tool call cleanup: {:?}", e);
                        }
                    }
                    changed = rx.changed() => {
                        match changed {
                            Ok(_) => break, // shutdown signal
                            Err(_) => break, // sender dropped
                        }
                    }
                }
            }
            info!("Stopped tool call cleanup task");
        });

        *handle_guard = Some(handle);
    }

    /// Stop the cleanup task and await it.
    pub async fn stop_cleanup_task(&self) {
        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _ = tx.send(true);
        }

        if let Some(handle) = self.cleanup_handle.lock().await.take() {
            if let Err(join_err) = handle.await {
                error!("Cleanup task join error: {:?}", join_err);
            }
        }
    }

    async fn cleanup_expired_tool_calls(&self, timeout: Duration) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = Utc::now();
        let mut map = self.inner.write().await;
        let before_len = map.len();
        let keys_to_remove: Vec<String> = map
            .iter()
            .filter_map(|(k, v)| {
                let age = now.signed_duration_since(v.created_at);
                // convert to std Duration and compare
                match age.to_std() {
                    Ok(d) if d > timeout => Some(k.clone()),
                    _ => None,
                }
            })
            .collect();

        for k in keys_to_remove {
            map.remove(&k);
        }

        if before_len != map.len() {
            info!("Cleaned up {} expired tool calls", before_len - map.len());
        }
        Ok(())
    }

    /// Clear all tool calls and stop background task.
    pub async fn cleanup_all(&self) {
        self.stop_cleanup_task().await;
        let mut map = self.inner.write().await;
        map.clear();
        info!("Cleaned up all tool calls");
    }
}
