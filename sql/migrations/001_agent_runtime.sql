CREATE TABLE IF NOT EXISTS agent_runs (
  run_id CHAR(36) PRIMARY KEY,
  conversation_id VARCHAR(128) NOT NULL,
  user_id VARCHAR(128) NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  user_goal TEXT NOT NULL,
  status VARCHAR(32) NOT NULL,
  current_step INT NOT NULL DEFAULT 0,
  max_steps INT NOT NULL,
  plan_json JSON NOT NULL,
  observations_json JSON NOT NULL,
  pending_approval_json JSON NULL,
  artifacts_json JSON NOT NULL,
  citations_json JSON NOT NULL,
  tool_call_count INT NOT NULL DEFAULT 0,
  max_tool_calls INT NOT NULL,
  error_json JSON NULL,
  final_answer MEDIUMTEXT NULL,
  version INT NOT NULL DEFAULT 1,
  created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ON UPDATE CURRENT_TIMESTAMP(6),
  KEY idx_agent_runs_scope (user_id, conversation_id, property_code),
  KEY idx_agent_runs_status_updated (status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS agent_steps (
  step_id CHAR(36) PRIMARY KEY,
  run_id CHAR(36) NOT NULL,
  step_number INT NOT NULL,
  step_type VARCHAR(64) NOT NULL,
  status VARCHAR(32) NOT NULL,
  input_json JSON NULL,
  output_json JSON NULL,
  error_json JSON NULL,
  started_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  completed_at TIMESTAMP(6) NULL,
  UNIQUE KEY uq_agent_steps_run_number (run_id, step_number),
  KEY idx_agent_steps_run_status (run_id, status),
  CONSTRAINT fk_agent_steps_run FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS tool_invocations (
  invocation_id CHAR(36) PRIMARY KEY,
  run_id CHAR(36) NOT NULL,
  step_id CHAR(36) NULL,
  tool_name VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL,
  attempt INT NOT NULL DEFAULT 1,
  idempotency_key VARCHAR(255) NULL,
  sanitized_input_json JSON NULL,
  output_json JSON NULL,
  citation_refs_json JSON NOT NULL,
  error_json JSON NULL,
  duration_ms INT NULL,
  started_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  completed_at TIMESTAMP(6) NULL,
  UNIQUE KEY uq_tool_invocation_idempotency (run_id, idempotency_key),
  KEY idx_tool_invocations_run_tool (run_id, tool_name),
  CONSTRAINT fk_tool_invocations_run FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_tool_invocations_step FOREIGN KEY (step_id) REFERENCES agent_steps(step_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS agent_checkpoints (
  checkpoint_id CHAR(36) PRIMARY KEY,
  run_id CHAR(36) NOT NULL,
  sequence_number INT NOT NULL,
  transition_name VARCHAR(64) NOT NULL,
  state_json JSON NOT NULL,
  created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  UNIQUE KEY uq_agent_checkpoints_sequence (run_id, sequence_number),
  KEY idx_agent_checkpoints_latest (run_id, created_at),
  CONSTRAINT fk_agent_checkpoints_run FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS agent_artifacts (
  artifact_id CHAR(36) PRIMARY KEY,
  run_id CHAR(36) NOT NULL,
  artifact_type VARCHAR(64) NOT NULL,
  name VARCHAR(255) NOT NULL,
  content_json JSON NULL,
  storage_uri VARCHAR(1024) NULL,
  content_hash CHAR(64) NULL,
  created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  KEY idx_agent_artifacts_run_type (run_id, artifact_type),
  CONSTRAINT fk_agent_artifacts_run FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS citation_evidence (
  citation_id CHAR(36) PRIMARY KEY,
  run_id CHAR(36) NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  source_type VARCHAR(64) NOT NULL,
  source_name VARCHAR(255) NOT NULL,
  tool_invocation_id CHAR(36) NULL,
  document_id VARCHAR(255) NULL,
  chunk_id VARCHAR(255) NULL,
  content_hash CHAR(64) NULL,
  source_url VARCHAR(2048) NULL,
  evidence_json JSON NULL,
  retrieved_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  index_version VARCHAR(128) NULL,
  KEY idx_citation_evidence_run (run_id),
  KEY idx_citation_evidence_scope (property_code, source_type),
  CONSTRAINT fk_citation_evidence_run FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_citation_evidence_tool FOREIGN KEY (tool_invocation_id)
    REFERENCES tool_invocations(invocation_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
