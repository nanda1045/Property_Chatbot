import { useEffect, useMemo, useRef, useState } from "react";
import type { AnchorHTMLAttributes, FormEvent, KeyboardEvent } from "react";
import {
  Bot,
  Building2,
  ExternalLink,
  Loader2,
  MapPin,
  MessageSquareText,
  Send,
  Sparkles
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ComponentRenderer } from "./components/ComponentRenderer";
import { executeApprovedSql, getModels, getProperties, sendChatStream } from "./lib/api";
import type { ChatTurn, ModelOption, PropertyOption, UIComponent } from "./types";

const STARTER_QUESTIONS = [
  "What is the latest occupancy and market rent?",
  "Does this property have EV charging?",
  "Show me the occupancy trend over time.",
  "Which units have the highest balances?"
];

const MARKDOWN_COMPONENTS = {
  a: ({ href, children }: AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  )
};

function displayProperty(property?: PropertyOption) {
  if (!property) {
    return "";
  }
  return `${property.property_code.toUpperCase()} - ${property.property_name}`;
}

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [properties, setProperties] = useState<PropertyOption[]>([]);
  const [model, setModel] = useState("anthropic:claude-haiku-4-5-20251001");
  const [propertyCode, setPropertyCode] = useState("115r");
  const [message, setMessage] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [loading, setLoading] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const turnListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function loadOptions() {
      try {
        const [modelResponse, propertyResponse] = await Promise.all([
          getModels(),
          getProperties()
        ]);
        setModels(modelResponse.models);
        setModel(modelResponse.default);
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
          message: question
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
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId ? { ...turn, response, streamedAnswer: undefined } : turn
        )
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

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitQuestion(message);
  }

  async function handleSqlApproval(turnId: string, component: UIComponent) {
    const data = component.data;
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      return;
    }
    const payload = data as Record<string, unknown>;
    const sql = typeof payload.sql === "string" ? payload.sql : "";
    const question = typeof payload.question === "string" ? payload.question : "";
    const approvedPropertyCode =
      typeof payload.property_code === "string" ? payload.property_code : propertyCode;
    const approvedModel = typeof payload.model === "string" ? payload.model : model;
    if (!sql || !question) {
      return;
    }

    setLoading(true);
    try {
      const response = await executeApprovedSql({
        propertyCode: approvedPropertyCode,
        model: approvedModel,
        sql,
        question
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
    } catch (error) {
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? {
                ...turn,
                error: error instanceof Error ? error.message : "SQL approval failed."
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
          {activeModel ? <span className="sr-only">Selected model: {activeModel.label}</span> : null}
        </div>

        {turns.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">
              <MessageSquareText aria-hidden="true" />
            </div>
            <h3>Ask about this property</h3>
            <p>
              Pull together rent-roll metrics, website evidence, charts, tables, and
              source links for the selected property.
            </p>
            <div className="starter-grid">
              {STARTER_QUESTIONS.map((question) => (
                <button key={question} type="button" onClick={() => void submitQuestion(question)}>
                  {question}
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

                    {turn.response && turn.response.components.length > 0 ? (
                      <div className="component-grid">
                        {turn.response.components.map((component, index) => (
                          <ComponentRenderer
                            key={`${component.type}-${component.title}-${index}`}
                            component={component}
                            onApprove={(approvalComponent) =>
                              void handleSqlApproval(turn.id, approvalComponent)
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
    </main>
  );
}
