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
  run_created: "Run created",
  planning_started: "Planning started",
  plan_created: "Plan created",
  step_started: "Step started",
  tool_started: "Tool started",
  tool_succeeded: "Tool succeeded",
  tool_failed: "Tool failed",
  tool_retried: "Tool retried",
  approval_requested: "SQL approval requested",
  approval_received: "SQL approval received",
  verification_started: "Verification started",
  verification_failed: "Verification failed",
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
  if (event.event_type === "tool_retried") {
    return <RotateCw aria-hidden="true" />;
  }
  if (event.event_type.includes("approval")) {
    return <ShieldCheck aria-hidden="true" />;
  }
  if (event.event_type.includes("verification")) {
    return <DatabaseZap aria-hidden="true" />;
  }
  if (event.event_type.endsWith("failed")) {
    return <XCircle aria-hidden="true" />;
  }
  if (event.event_type === "run_cancelled") {
    return <Ban aria-hidden="true" />;
  }
  if (event.event_type.endsWith("completed") || event.event_type.endsWith("succeeded")) {
    return <CheckCircle2 aria-hidden="true" />;
  }
  return <Activity aria-hidden="true" />;
}

function eventTone(event: AgentRunEvent) {
  if (event.event_type.endsWith("failed")) {
    return "failed";
  }
  if (event.event_type === "run_cancelled") {
    return "cancelled";
  }
  if (event.event_type === "tool_retried") {
    return "retried";
  }
  if (event.event_type.endsWith("completed") || event.event_type.endsWith("succeeded")) {
    return "succeeded";
  }
  return "active";
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
                    <span>{citation.source_type.replaceAll("_", " ")}</span>
                    <strong>{citation.source_name}</strong>
                    <code>{citation.citation_id.slice(0, 8)}</code>
                  </summary>
                  <dl>
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
                  <details className="trace-details">
                    <summary>Query and evidence</summary>
                    <pre>{JSON.stringify({
                      query_parameters: citation.query_parameters,
                      evidence: citation.evidence
                    }, null, 2)}</pre>
                  </details>
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
                  <strong>{EVENT_LABELS[event.event_type] ?? event.event_type}</strong>
                  {event.duration_ms !== null && event.duration_ms !== undefined ? (
                    <time>{formatDuration(event.duration_ms)}</time>
                  ) : null}
                </div>
                <p>
                  {event.tool_name ? <span>{event.tool_name}</span> : null}
                  {event.attempt ? <span>attempt {event.attempt}</span> : null}
                  {event.error_type ? <span>{event.error_type}</span> : null}
                </p>
                {Object.keys(event.payload).length > 0 ? (
                  <details className="trace-details">
                    <summary>Operational details</summary>
                    <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                  </details>
                ) : null}
              </div>
            </li>
          ))}
        </ol>
        {error ? <p className="trace-error">Latest refresh failed: {error}</p> : null}
      </div>
    </details>
  );
}
