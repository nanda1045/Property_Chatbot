CREATE TABLE IF NOT EXISTS conversation_threads (
  thread_id CHAR(36) PRIMARY KEY,
  user_id VARCHAR(128) NOT NULL,
  conversation_id VARCHAR(128) NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  summary_text MEDIUMTEXT NULL,
  summarized_through BIGINT UNSIGNED NOT NULL DEFAULT 0,
  created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ON UPDATE CURRENT_TIMESTAMP(6),
  UNIQUE KEY uq_conversation_threads_scope (user_id, conversation_id, property_code),
  KEY idx_conversation_threads_updated (user_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS conversation_turns (
  sequence_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  turn_id CHAR(36) NOT NULL,
  thread_id CHAR(36) NOT NULL,
  run_id CHAR(36) NULL,
  user_message TEXT NOT NULL,
  assistant_answer MEDIUMTEXT NOT NULL,
  tool_result_keys_json JSON NOT NULL,
  component_types_json JSON NOT NULL,
  created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  UNIQUE KEY uq_conversation_turns_id (turn_id),
  KEY idx_conversation_turns_thread_sequence (thread_id, sequence_id),
  KEY idx_conversation_turns_run (run_id),
  CONSTRAINT fk_conversation_turns_thread FOREIGN KEY (thread_id)
    REFERENCES conversation_threads(thread_id) ON DELETE CASCADE,
  CONSTRAINT fk_conversation_turns_run FOREIGN KEY (run_id)
    REFERENCES agent_runs(run_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
