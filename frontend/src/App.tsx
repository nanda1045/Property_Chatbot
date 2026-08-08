import { useEffect, useMemo, useRef, useState } from "react";
import type { AnchorHTMLAttributes, FormEvent, KeyboardEvent } from "react";
import {
  Bot,
  Building2,
  CheckCircle2,
  ExternalLink,
  GitPullRequestArrow,
  Info,
  LockKeyhole,
  Loader2,
  LogIn,
  LogOut,
  MapPin,
  MessageSquareText,
  Send,
  ShieldCheck,
  Sparkles,
  Workflow,
  X
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ComponentRenderer } from "./components/ComponentRenderer";
import { RunTracePanel } from "./components/RunTracePanel";
import { useAuth } from "./auth";
import {
  approveAgentRun,
  cancelAgentRun,
  getAgentRunTrace,
  getModels,
  getProperties,
  sendChatStream
} from "./lib/api";
import type {
  ChatResponse,
  ChatTurn,
  DemoRole,
  ModelOption,
  PropertyOption,
  UIComponent
} from "./types";

const STARTER_WORKFLOWS = [
  {
    label: "Retrieval",
    question: "What amenities are available at this property?"
  },
  {
    label: "Structured analytics",
    question: "Show the occupancy trend and explain any recent decline."
  },
  {
    label: "Hybrid investigation",
    question:
      "Investigate why occupancy declined and give me an executive summary with supporting evidence."
  },
  {
    label: "Privileged action",
    question:
      "Calculate the average outstanding balance for occupied units with balances above the property's overall average."
  }
];

const MARKDOWN_COMPONENTS = {
  a: ({ href, children }: AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  )
};

const DEFAULT_MODEL_ID = "anthropic:claude-haiku-4-5-20251001";
const CONVERSATION_ID_STORAGE_KEY = "aker_conversation_id";

function displayProperty(property?: PropertyOption) {
  if (!property) {
    return "";
  }
  return `${property.property_code.toUpperCase()} - ${property.property_name}`;
}

function createConversationId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function getInitialConversationId() {
  const stored = window.localStorage.getItem(CONVERSATION_ID_STORAGE_KEY);
  if (stored) {
    return stored;
  }

  const next = createConversationId();
  window.localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, next);
  return next;
}

function AuthenticatedApp() {
  const { mode: authMode, user, signOut, switchDemoIdentity } = useAuth();
  const [models, setModels] = useState<ModelOption[]>([]);
  const [properties, setProperties] = useState<PropertyOption[]>([]);
  const [model, setModel] = useState(DEFAULT_MODEL_ID);
  const [propertyCode, setPropertyCode] = useState("115r");
  const [conversationId, setConversationId] = useState(getInitialConversationId);
  const [message, setMessage] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [loading, setLoading] = useState(false);
  const [cancellingRunId, setCancellingRunId] = useState<string | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const turnListRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    async function loadOptions() {
      try {
        const [modelResponse, propertyResponse] = await Promise.all([
          getModels(),
          getProperties()
        ]);
        setModels(modelResponse.models);
        setModel(
          modelResponse.models.some((option) => option.id === DEFAULT_MODEL_ID)
            ? DEFAULT_MODEL_ID
            : modelResponse.default
        );
        setProperties(propertyResponse.properties);
        if (propertyResponse.properties[0]) {
          setPropertyCode(propertyResponse.properties[0].property_code);
        }
      } catch (error) {
        setBootError(error instanceof Error ? error.message : "Unable to load options.");
      }
    }

    void loadOptions();
  }, []);

  const activeProperty = useMemo(
    () => properties.find((property) => property.property_code === propertyCode),
    [properties, propertyCode]
  );
  const activeModel = useMemo(
    () => models.find((option) => option.id === model),
    [model, models]
  );

  useEffect(() => {
    turnListRef.current?.lastElementChild?.scrollIntoView({
      behavior: "smooth",
      block: "end"
    });
  }, [turns, loading]);

  useEffect(() => {
    if (!aboutOpen) {
      return;
    }
    function closeOnEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        setAboutOpen(false);
      }
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [aboutOpen]);

  async function loadRunTrace(
    turnId: string,
    response: ChatResponse,
    scopedConversationId: string
  ) {
    if (!response.run_id) {
      return;
    }
    setTurns((current) =>
      current.map((turn) =>
        turn.id === turnId
          ? { ...turn, traceLoading: true, traceError: undefined }
          : turn
      )
    );
    try {
      const trace = await getAgentRunTrace({
        runId: response.run_id,
        propertyCode: response.property_code,
        conversationId: scopedConversationId
      });
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? { ...turn, trace, traceLoading: false, traceError: undefined }
            : turn
        )
      );
    } catch (error) {
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                traceLoading: false,
                traceError: error instanceof Error ? error.message : "Unable to load run trace."
              }
            : turn
        )
      );
    }
  }

  async function submitQuestion(rawQuestion: string) {
    const question = rawQuestion.trim();
    if (!question || loading) {
      return;
    }

    const turnId = crypto.randomUUID();
    setMessage("");
    setLoading(true);
    setTurns((current) => [...current, { id: turnId, question }]);

    try {
      const response = await sendChatStream(
        {
          propertyCode,
          model,
          message: question,
          conversationId
        },
        {
          onToken: (token) => {
            setTurns((current) =>
              current.map((turn) =>
                turn.id === turnId
                  ? { ...turn, streamedAnswer: `${turn.streamedAnswer ?? ""}${token}` }
                  : turn
              )
            );
          }
        }
      );
      if (response.conversation_id && response.conversation_id !== conversationId) {
        setConversationId(response.conversation_id);
        window.localStorage.setItem(CONVERSATION_ID_STORAGE_KEY, response.conversation_id);
      }
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId ? { ...turn, response, streamedAnswer: undefined } : turn
        )
      );
      void loadRunTrace(
        turnId,
        response,
        response.conversation_id ?? conversationId
      );
    } catch (error) {
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                error: error instanceof Error ? error.message : "Request failed."
              }
            : turn
        )
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleRunCancel(turnId: string) {
    const turn = turns.find((candidate) => candidate.id === turnId);
    const response = turn?.response;
    if (!response?.run_id) {
      return;
    }
    const scopedConversationId = response.conversation_id ?? conversationId;
    setCancellingRunId(response.run_id);
    try {
      const run = await cancelAgentRun({
        runId: response.run_id,
        propertyCode: response.property_code,
        conversationId: scopedConversationId
      });
      setTurns((current) =>
        current.map((candidate) =>
          candidate.id === turnId
            ? {
                ...candidate,
                response: candidate.response
                  ? { ...candidate.response, run_status: run.status }
                  : candidate.response
              }
            : candidate
        )
      );
      await loadRunTrace(turnId, response, scopedConversationId);
    } catch (error) {
      setTurns((current) =>
        current.map((candidate) =>
          candidate.id === turnId
            ? {
                ...candidate,
                traceError: error instanceof Error ? error.message : "Unable to cancel run."
              }
            : candidate
        )
      );
    } finally {
      setCancellingRunId(null);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitQuestion(message);
  }

  async function handleSqlApproval(
    turnId: string,
    component: UIComponent,
    approved: boolean
  ) {
    const data = component.data;
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      return;
    }
    const payload = data as Record<string, unknown>;
    const turn = turns.find((candidate) => candidate.id === turnId);
    const runId =
      typeof payload.run_id === "string" ? payload.run_id : turn?.response?.run_id ?? "";
    const approvedPropertyCode =
      typeof payload.property_code === "string" ? payload.property_code : propertyCode;
    if (!runId) {
      return;
    }

    setLoading(true);
    try {
      const response = await approveAgentRun({
        runId,
        propertyCode: approvedPropertyCode,
        conversationId,
        approved
      });
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                response: {
                  ...response,
                  answer_markdown: `${turn.response?.answer_markdown ?? ""}\n\n${response.answer_markdown}`
                }
              }
            : turn
        )
      );
      void loadRunTrace(
        turnId,
        response,
        response.conversation_id ?? conversationId
      );
    } catch (error) {
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                error: error instanceof Error ? error.message : "SQL decision failed."
              }
            : turn
        )
      );
    } finally {
      setLoading(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion(message);
    }
  }

  function selectStarterWorkflow(question: string) {
    setMessage(question);
    window.requestAnimationFrame(() => {
      composerRef.current?.focus();
      composerRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  return (
    <main className="app-shell">
      <aside className="control-rail">
        <div className="brand-lockup">
          <Building2 aria-hidden="true" />
          <div>
            <h1>Aker Assistant</h1>
            <p>Property-scoped AI workspace</p>
          </div>
        </div>

        {user ? (
          <section className="user-control" aria-label="Signed-in user">
            <div>
              <strong>{user.display_name}</strong>
              <span>
                {user.role ?? "No application role"} • Property {propertyCode.toUpperCase()}
              </span>
              <small>{authMode === "entra" ? "Microsoft Entra ID" : "Local demo identity"}</small>
            </div>
            {authMode === "entra" ? (
              <button type="button" onClick={() => void signOut()} aria-label="Sign out">
                <LogOut aria-hidden="true" />
              </button>
            ) : null}
          </section>
        ) : null}

        {authMode === "local" && user && switchDemoIdentity ? (
          <label className="demo-identity-switcher">
            <span>Local demo identity</span>
            <select
              value={user.role ?? "Viewer"}
              onChange={(event) =>
                void switchDemoIdentity(event.target.value as DemoRole)
              }
            >
              <option value="Viewer">Demo Viewer</option>
              <option value="Analyst">Demo Analyst</option>
              <option value="PropertyManager">Demo Property Manager</option>
            </select>
            <small>Local mode only • backend-issued identity</small>
          </label>
        ) : null}

        <div className="control-card">
          <label>
            <span>Property</span>
            <select value={propertyCode} onChange={(event) => setPropertyCode(event.target.value)}>
              {properties.map((property) => (
                <option key={property.property_code} value={property.property_code}>
                  {displayProperty(property)}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Model</span>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {models.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {activeProperty ? (
          <section className="property-summary">
            <strong>{activeProperty.property_name}</strong>
            {activeProperty.address ? (
              <span>
                <MapPin aria-hidden="true" />
                {activeProperty.address}
              </span>
            ) : null}
            {activeProperty.source_site ? (
              <a href={activeProperty.source_site} target="_blank" rel="noreferrer">
                Property website
                <ExternalLink aria-hidden="true" />
              </a>
            ) : null}
          </section>
        ) : null}

        <section className="security-summary" aria-label="Agent security status">
          <div className="security-summary-heading">
            <ShieldCheck aria-hidden="true" />
            <span>Security &amp; agent status</span>
          </div>
          <ul>
            <li>
              <CheckCircle2 aria-hidden="true" />
              <span>
                <strong>Authenticated</strong>
                <small>{authMode === "entra" ? "Entra token validated" : "Explicit local demo mode"}</small>
              </span>
            </li>
            <li>
              <LockKeyhole aria-hidden="true" />
              <span>
                <strong>Property scoped</strong>
                <small>Backend enforced • {propertyCode.toUpperCase()}</small>
              </span>
            </li>
            <li>
              <Workflow aria-hidden="true" />
              <span>
                <strong>Bounded agent</strong>
                <small>Step, tool, retry &amp; time limits</small>
              </span>
            </li>
            <li>
              <GitPullRequestArrow aria-hidden="true" />
              <span>
                <strong>Human approval enforced</strong>
                <small>Sensitive SQL is checkpoint gated</small>
              </span>
            </li>
          </ul>
        </section>

        {bootError ? <p className="status-error">{bootError}</p> : null}
      </aside>

      <section className="workspace">
        <div className="chat-header">
          <div>
            <span className="eyebrow">
              <Sparkles aria-hidden="true" />
              {propertyCode.toUpperCase()}
            </span>
            <h2>{activeProperty?.property_name ?? "Property Assistant"}</h2>
          </div>
          <div className="header-actions">
            {activeModel ? <span className="sr-only">Selected model: {activeModel.label}</span> : null}
            <button type="button" className="about-trigger" onClick={() => setAboutOpen(true)}>
              <Info aria-hidden="true" />
              How this agent works
            </button>
          </div>
        </div>

        {turns.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">
              <MessageSquareText aria-hidden="true" />
            </div>
            <h3>Try an agent workflow</h3>
            <p>
              Choose an example to place it in the composer. Review or edit it before
              asking the property-scoped agent to run.
            </p>
            <div className="starter-grid">
              {STARTER_WORKFLOWS.map((workflow) => (
                <button
                  key={workflow.label}
                  type="button"
                  onClick={() => selectStarterWorkflow(workflow.question)}
                >
                  <span>{workflow.label}</span>
                  <strong>{workflow.question}</strong>
                </button>
              ))}
            </div>
          </div>
        ) : null}

        <div className="turn-list" ref={turnListRef}>
          {turns.map((turn) => (
            <article className="turn" key={turn.id}>
              <div className="question-row">
                <span>You</span>
                <p>{turn.question}</p>
              </div>

              {turn.error ? <p className="status-error">{turn.error}</p> : null}

              {turn.response || turn.streamedAnswer ? (
                <div className="assistant-message">
                  <div className="assistant-avatar">
                    <Bot aria-hidden="true" />
                  </div>
                  <div className="answer-block">
                    <div className="assistant-label">Assistant</div>
                    <div className="markdown-body">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={MARKDOWN_COMPONENTS}
                      >
                        {turn.response?.answer_markdown ?? turn.streamedAnswer ?? ""}
                      </ReactMarkdown>
                      {!turn.response ? <span className="stream-cursor" aria-hidden="true" /> : null}
                    </div>

                    {turn.response && turn.response.citation_ids.length > 0 ? (
                      <div className="answer-citations" aria-label="Answer evidence">
                        <span>Evidence</span>
                        {turn.response.citation_ids.map((citationId) => (
                          <code key={citationId}>{citationId.slice(0, 8)}</code>
                        ))}
                      </div>
                    ) : null}

                    {turn.response && turn.response.components.length > 0 ? (
                      <div className="component-grid">
                        {turn.response.components.map((component, index) => (
                          <ComponentRenderer
                            key={`${component.type}-${component.title}-${index}`}
                            component={component}
                            onApprove={(approvalComponent) =>
                              void handleSqlApproval(turn.id, approvalComponent, true)
                            }
                            onReject={(approvalComponent) =>
                              void handleSqlApproval(turn.id, approvalComponent, false)
                            }
                          />
                        ))}
                      </div>
                    ) : null}

                    {turn.response && turn.response.sources.length > 0 ? (
                      <div className="source-list">
                        <h3>Sources</h3>
                        {turn.response.sources.map((source, index) => (
                          <a
                            key={`${source.source_url}-${index}`}
                            href={source.source_url ?? "#"}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <span>{source.page_type ?? source.tool ?? "source"}</span>
                            <strong>{source.title ?? source.source_url}</strong>
                            <ExternalLink aria-hidden="true" />
                          </a>
                        ))}
                      </div>
                    ) : null}

                    {turn.response?.run_id ? (
                      <RunTracePanel
                        trace={turn.trace}
                        loading={turn.traceLoading}
                        error={turn.traceError}
                        cancelling={cancellingRunId === turn.response.run_id}
                        onCancel={() => void handleRunCancel(turn.id)}
                      />
                    ) : null}
                  </div>
                </div>
              ) : loading ? (
                <div className="assistant-message pending">
                  <div className="assistant-avatar">
                    <Bot aria-hidden="true" />
                  </div>
                  <div className="typing-bubble">
                    <Loader2 className="spin" aria-hidden="true" />
                    <span>Thinking</span>
                    <i />
                    <i />
                    <i />
                  </div>
                </div>
              ) : null}
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            ref={composerRef}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Ask about rent roll KPIs, charges, vacancies, amenities, or website content."
            rows={3}
          />
          <button type="submit" disabled={loading || !message.trim()} aria-label="Ask assistant">
            {loading ? <Loader2 className="spin" aria-hidden="true" /> : <Send aria-hidden="true" />}
            Ask
          </button>
        </form>
      </section>

      {aboutOpen ? (
        <div className="about-backdrop" role="presentation" onMouseDown={() => setAboutOpen(false)}>
          <section
            className="about-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="about-agent-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="about-dialog-heading">
              <div>
                <span className="eyebrow">
                  <ShieldCheck aria-hidden="true" />
                  Identity-aware orchestration
                </span>
                <h2 id="about-agent-title">How this agent works</h2>
              </div>
              <button type="button" onClick={() => setAboutOpen(false)} aria-label="Close">
                <X aria-hidden="true" />
              </button>
            </div>

            <div className="agent-flow" aria-label="Agent architecture flow">
              {[
                "User",
                authMode === "entra" ? "Entra authentication" : "Local demo authentication",
                "FastAPI",
                "Bounded agent runtime",
                "Authorization policy",
                "Typed tool registry",
                "MySQL + hybrid retrieval",
                "Evidence verification",
                "Response"
              ].map((step, index, steps) => (
                <div className="agent-flow-item" key={step}>
                  <span>{step}</span>
                  {index < steps.length - 1 ? <strong aria-hidden="true">→</strong> : null}
                </div>
              ))}
            </div>

            <ul className="agent-guardrails">
              <li>Tools are typed and allowlisted.</li>
              <li>Identity and property scope are injected and enforced by the backend.</li>
              <li>Sensitive SQL pauses for explicit, re-authorized human approval.</li>
              <li>Runs are checkpointed with bounded steps, retries, tools, and duration.</li>
              <li>Responses are verified against stored, property-scoped evidence.</li>
              <li>Run Trace exposes operational decisions—not private model reasoning.</li>
            </ul>
          </section>
        </div>
      ) : null}
    </main>
  );
}

export default function App() {
  const { mode, user, loading, error, signIn } = useAuth();

  if (loading) {
    return (
      <div className="auth-gate">
        <Loader2 className="spin" aria-hidden="true" />
        <p>Loading your workspace…</p>
      </div>
    );
  }
  if (!user) {
    return (
      <div className="auth-gate">
        <Building2 aria-hidden="true" />
        <h1>Aker Assistant</h1>
        <p>{error ?? "Sign in with your Microsoft work account to continue."}</p>
        {mode === "entra" ? (
          <button type="button" onClick={() => void signIn()}>
            <LogIn aria-hidden="true" />
            Sign in with Microsoft
          </button>
        ) : null}
      </div>
    );
  }
  return <AuthenticatedApp />;
}
