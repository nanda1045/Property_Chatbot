CREATE TABLE IF NOT EXISTS agent_events (
  sequence_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  event_id CHAR(36) NOT NULL,
  run_id CHAR(36) NOT NULL,
  event_type VARCHAR(64) NOT NULL,
  conversation_id VARCHAR(128) NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  step_id CHAR(36) NULL,
  tool_name VARCHAR(128) NULL,
  attempt INT NULL,
  duration_ms INT NULL,
  event_timestamp TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  error_type VARCHAR(128) NULL,
  payload_json JSON NULL,
  UNIQUE KEY uq_agent_events_id (event_id),
  KEY idx_agent_events_run_sequence (run_id, sequence_id),
  KEY idx_agent_events_scope (conversation_id, property_code, event_timestamp),
  CONSTRAINT fk_agent_events_run FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_agent_events_step FOREIGN KEY (step_id) REFERENCES agent_steps(step_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
