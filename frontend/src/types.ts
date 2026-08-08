export type ModelOption = {
  id: string;
  label: string;
  provider: string;
};

export type AuthenticatedUser = {
  user_id: string;
  display_name: string;
  email?: string | null;
  tenant_id: string;
  roles: string[];
  role?: string | null;
};

export type DemoRole = "Viewer" | "Analyst" | "PropertyManager";

export type PropertyOption = {
  property_code: string;
  property_name: string;
  address?: string | null;
  source_site?: string | null;
};

export type UIComponent = {
  type: string;
  title: string;
  data: unknown;
  description?: string | null;
};

export type Source = {
  property_code: string;
  title?: string | null;
  source_url?: string | null;
  page_type?: string | null;
  tool?: string | null;
};

export type ChatResponse = {
  property_code: string;
  model: string;
  conversation_id?: string | null;
  run_id?: string | null;
  run_status?: string | null;
  answer_markdown: string;
  components: UIComponent[];
  sources: Source[];
  citation_ids: string[];
  tool_results: Record<string, unknown>;
};

export type AgentRunDetail = {
  run_id: string;
  conversation_id: string;
  property_code: string;
  user_goal: string;
  status: string;
  current_step: number;
  max_steps: number;
  plan: Array<Record<string, unknown>>;
  pending_approval?: Record<string, unknown> | null;
  tool_call_count: number;
  max_tool_calls: number;
  error?: Record<string, unknown> | null;
  final_answer?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AgentRunStep = {
  step_id: string;
  run_id: string;
  step_number: number;
  step_type: string;
  status: string;
  input?: Record<string, unknown> | null;
  output?: Record<string, unknown> | null;
  error?: Record<string, unknown> | null;
  started_at?: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
};

export type AgentRunEvent = {
  event_id: string;
  run_id: string;
  event_type: string;
  conversation_id: string;
  property_code: string;
  step_id?: string | null;
  tool_name?: string | null;
  attempt?: number | null;
  duration_ms?: number | null;
  timestamp: string;
  error_type?: string | null;
  payload: Record<string, unknown>;
};

export type AgentRunCitation = {
  citation_id: string;
  run_id: string;
  property_code: string;
  source_type: "structured_tool" | "retrieval";
  source_name: string;
  tool_invocation_id?: string | null;
  query_parameters: Record<string, unknown>;
  data_timestamp?: string | null;
  document_id?: string | null;
  chunk_id?: string | null;
  content_hash: string;
  source_url?: string | null;
  evidence: Record<string, unknown>;
  retrieved_at: string;
  index_version?: string | null;
};

export type AgentRunTrace = {
  run: AgentRunDetail;
  steps: AgentRunStep[];
  events: AgentRunEvent[];
  citations: AgentRunCitation[];
};

export type ChatTurn = {
  id: string;
  question: string;
  streamedAnswer?: string;
  response?: ChatResponse;
  trace?: AgentRunTrace;
  traceLoading?: boolean;
  traceError?: string;
  error?: string;
};
