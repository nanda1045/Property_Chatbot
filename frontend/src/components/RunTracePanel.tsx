import {
  Activity,
  Ban,
  BookOpenCheck,
  CheckCircle2,
  ChevronRight,
  Clock3,
  DatabaseZap,
  RotateCw,
  ShieldCheck,
  XCircle
} from "lucide-react";

import type { AgentRunEvent, AgentRunTrace } from "../types";

type RunTracePanelProps = {
  trace?: AgentRunTrace;
  loading?: boolean;
  error?: string;
  cancelling?: boolean;
  onCancel: () => void;
};

const EVENT_LABELS: Record<string, string> = {
  run_created: "Run checkpoint created",
  AUTHENTICATED: "Request authenticated",
  AUTHORIZATION_ALLOWED: "Authorization allowed",
  AUTHORIZATION_DENIED: "Authorization denied",
  SQL_APPROVAL_AUTHORIZED: "SQL approval authorization allowed",
  SQL_APPROVAL_DENIED: "SQL approval authorization denied",
  planning_started: "Planning started",
  plan_created: "Plan created",
  step_started: "Execution step started",
  tool_started: "Tool requested",
  tool_succeeded: "Tool execution completed",
  tool_failed: "Tool execution failed",
  tool_retried: "Tool execution retried",
  approval_requested: "SQL approval required",
  approval_received: "SQL approval decision received",
  verification_started: "Evidence verification started",
  verification_failed: "Evidence verification failed",
  run_completed: "Run completed",
  run_failed: "Run failed",
  run_cancelled: "Run cancelled"
};

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

function formatDuration(duration?: number | null) {
  if (duration === null || duration === undefined) {
    return "—";
  }
  if (duration < 1000) {
    return `${duration} ms`;
  }
  return `${(duration / 1000).toFixed(duration < 10_000 ? 2 : 1)} s`;
}

function eventIcon(event: AgentRunEvent) {
  const eventType = event.event_type.toLowerCase();
  if (eventType === "tool_retried") {
    return <RotateCw aria-hidden="true" />;
  }
  if (eventType.includes("authorization") || eventType.includes("authenticated")) {
    return <ShieldCheck aria-hidden="true" />;
  }
  if (eventType.includes("approval")) {
    return <ShieldCheck aria-hidden="true" />;
  }
  if (eventType.includes("verification")) {
    return <DatabaseZap aria-hidden="true" />;
  }
  if (eventType.endsWith("failed") || eventType.endsWith("denied")) {
    return <XCircle aria-hidden="true" />;
  }
  if (eventType === "run_cancelled") {
    return <Ban aria-hidden="true" />;
  }
  if (
    eventType.endsWith("completed") ||
    eventType.endsWith("succeeded") ||
    eventType.endsWith("allowed")
  ) {
    return <CheckCircle2 aria-hidden="true" />;
  }
  return <Activity aria-hidden="true" />;
}

function eventTone(event: AgentRunEvent) {
  const eventType = event.event_type.toLowerCase();
  if (eventType.endsWith("failed") || eventType.endsWith("denied")) {
    return "failed";
  }
  if (eventType === "run_cancelled") {
    return "cancelled";
  }
  if (eventType === "tool_retried") {
    return "retried";
  }
  if (
    eventType.endsWith("completed") ||
    eventType.endsWith("succeeded") ||
    eventType.endsWith("allowed") ||
    eventType === "authenticated"
  ) {
    return "succeeded";
  }
  return "active";
}

function eventLabel(event: AgentRunEvent) {
  if (event.event_type === "approval_received") {
    const decision = event.payload.decision;
    if (decision === "approved") {
      return "User approved SQL";
    }
    if (decision === "rejected") {
      return "User rejected SQL";
    }
  }
  if (event.tool_name === "execute_approved_sql") {
    if (event.event_type === "tool_started") {
      return "Approved SQL execution started";
    }
    if (event.event_type === "tool_succeeded") {
      return "Approved SQL execution completed";
    }
    if (event.event_type === "tool_failed") {
      return "Approved SQL execution failed";
    }
  }
  return EVENT_LABELS[event.event_type] ?? event.event_type.replaceAll("_", " ");
}

function eventCategory(event: AgentRunEvent) {
  const eventType = event.event_type.toLowerCase();
  if (eventType.includes("auth")) {
    return "Security";
  }
  if (eventType.includes("approval")) {
    return "Human approval";
  }
  if (eventType.includes("tool") || event.tool_name) {
    return event.tool_name === "execute_approved_sql" ? "SQL execution" : "Tool execution";
  }
  if (eventType.includes("planning") || eventType.includes("plan_")) {
    return "Planning";
  }
  if (eventType.includes("verification")) {
    return "Verification";
  }
  if (eventType.startsWith("run_")) {
    return "Run lifecycle";
  }
  return "Execution";
}

function safeEventMetadata(event: AgentRunEvent) {
  const entries: Array<[string, string]> = [["Property", event.property_code.toUpperCase()]];
  const scalarFields: Array<[string, string]> = [
    ["Permission", "permission"],
    ["Decision", "decision"],
    ["Outcome", "outcome"],
    ["Role", "role"],
    ["Status", "status"],
    ["Step", "step_type"],
    ["Reason", "reason"]
  ];

  for (const [label, key] of scalarFields) {
    const value = event.payload[key];
    if (typeof value === "string" && value.trim()) {
      entries.push([label, value.replaceAll("_", " ")]);
    }
  }

  const outputSummary = event.payload.output_summary;
  if (outputSummary && typeof outputSummary === "object" && !Array.isArray(outputSummary)) {
    const summary = outputSummary as Record<string, unknown>;
    const count = summary.row_count ?? summary.item_count;
    if (typeof count === "number") {
      entries.push(["Records", String(count)]);
    }
  }

  return entries;
}

export function RunTracePanel({
  trace,
  loading = false,
  error,
  cancelling = false,
  onCancel
}: RunTracePanelProps) {
  if (loading && !trace) {
    return <p className="trace-loading">Loading operational run trace…</p>;
  }
  if (error && !trace) {
    return <p className="trace-error">Run trace unavailable: {error}</p>;
  }
  if (!trace) {
    return null;
  }

  const finalEvent = [...trace.events]
    .reverse()
    .find((event) => ["run_completed", "run_failed", "run_cancelled"].includes(event.event_type));
  const canCancel = !TERMINAL_STATUSES.has(trace.run.status);

  return (
    <details className="run-trace-panel">
      <summary>
        <span className={`trace-status trace-status-${trace.run.status}`}>
          <Activity aria-hidden="true" />
          Run trace
        </span>
        <span>{trace.events.length} events</span>
        <span>{formatDuration(finalEvent?.duration_ms)}</span>
        <ChevronRight className="trace-chevron" aria-hidden="true" />
      </summary>

      <div className="trace-content">
        <div className="trace-overview">
          <div>
            <span>Status</span>
            <strong>{trace.run.status.replaceAll("_", " ")}</strong>
          </div>
          <div>
            <span>Steps</span>
            <strong>
              {trace.run.current_step}/{trace.run.max_steps}
            </strong>
          </div>
          <div>
            <span>Tool calls</span>
            <strong>
              {trace.run.tool_call_count}/{trace.run.max_tool_calls}
            </strong>
          </div>
          <div>
            <span>Property</span>
            <strong>{trace.run.property_code.toUpperCase()}</strong>
          </div>
          {canCancel ? (
            <button type="button" className="trace-cancel" onClick={onCancel} disabled={cancelling}>
              <Ban aria-hidden="true" />
              {cancelling ? "Cancelling…" : "Cancel run"}
            </button>
          ) : null}
        </div>

        {trace.steps.length > 0 ? (
          <div className="trace-step-grid">
            {trace.steps.map((step) => (
              <div key={step.step_id} className="trace-step">
                <Clock3 aria-hidden="true" />
                <span>Step {step.step_number}</span>
                <strong>{step.step_type.replaceAll("_", " ")}</strong>
                <time>{formatDuration(step.duration_ms)}</time>
              </div>
            ))}
          </div>
        ) : null}

        {trace.citations.length > 0 ? (
          <section className="trace-citations" aria-label="Stored evidence">
            <h4>
              <BookOpenCheck aria-hidden="true" />
              Stored evidence ({trace.citations.length})
            </h4>
            <div className="trace-citation-grid">
              {trace.citations.map((citation) => (
                <details key={citation.citation_id} className="trace-citation">
                  <summary>
                    <span>Evidence recorded</span>
                    <strong>{citation.source_name}</strong>
                    <code>{citation.citation_id.slice(0, 8)}</code>
                  </summary>
                  <dl>
                    <dt>Evidence type</dt>
                    <dd>{citation.source_type.replaceAll("_", " ")}</dd>
                    <dt>Property</dt>
                    <dd>{citation.property_code.toUpperCase()}</dd>
                    {citation.tool_invocation_id ? (
                      <>
                        <dt>Tool invocation</dt>
                        <dd><code>{citation.tool_invocation_id}</code></dd>
                      </>
                    ) : null}
                    {citation.document_id ? (
                      <>
                        <dt>Document</dt>
                        <dd><code>{citation.document_id}</code></dd>
                      </>
                    ) : null}
                    {citation.chunk_id ? (
                      <>
                        <dt>Chunk</dt>
                        <dd><code>{citation.chunk_id}</code></dd>
                      </>
                    ) : null}
                    <dt>Content hash</dt>
                    <dd><code>{citation.content_hash}</code></dd>
                    <dt>Retrieved</dt>
                    <dd>{citation.retrieved_at}</dd>
                    {citation.data_timestamp ? (
                      <>
                        <dt>Data timestamp</dt>
                        <dd>{citation.data_timestamp}</dd>
                      </>
                    ) : null}
                    {citation.index_version ? (
                      <>
                        <dt>Index version</dt>
                        <dd>{citation.index_version}</dd>
                      </>
                    ) : null}
                  </dl>
                  {citation.source_url ? (
                    <a href={citation.source_url} target="_blank" rel="noreferrer">
                      Open source
                    </a>
                  ) : null}
                </details>
              ))}
            </div>
          </section>
        ) : null}

        <ol className="trace-timeline">
          {trace.events.map((event) => (
            <li key={event.event_id} className={`trace-event trace-event-${eventTone(event)}`}>
              <span className="trace-event-icon">{eventIcon(event)}</span>
              <div>
                <div className="trace-event-heading">
                  <div>
                    <span className="trace-event-category">{eventCategory(event)}</span>
                    <strong>{eventLabel(event)}</strong>
                  </div>
                  {event.duration_ms !== null && event.duration_ms !== undefined ? (
                    <time>{formatDuration(event.duration_ms)}</time>
                  ) : null}
                </div>
                <p>
                  {event.tool_name ? <span>{event.tool_name}</span> : null}
                  {event.attempt ? <span>attempt {event.attempt}</span> : null}
                  {event.error_type ? <span>{event.error_type}</span> : null}
                </p>
                <dl className="trace-metadata">
                  {safeEventMetadata(event).map(([label, value]) => (
                    <div key={`${label}-${value}`}>
                      <dt>{label}</dt>
                      <dd>{value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </li>
          ))}
        </ol>
        {error ? <p className="trace-error">Latest refresh failed: {error}</p> : null}
      </div>
    </details>
  );
}
