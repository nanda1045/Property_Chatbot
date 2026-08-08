import type {
  AgentRunDetail,
  AgentRunCitation,
  AgentRunEvent,
  AgentRunStep,
  AgentRunTrace,
  AuthenticatedUser,
  ChatResponse,
  ModelOption,
  PropertyOption
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
type AccessTokenProvider = () => Promise<string | null>;
let accessTokenProvider: AccessTokenProvider = async () => null;

export function setAccessTokenProvider(provider: AccessTokenProvider) {
  accessTokenProvider = provider;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const accessToken = await accessTokenProvider();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      ...init?.headers
    },
    ...init
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function getAuthenticatedUser(): Promise<AuthenticatedUser> {
  return request("/auth/me");
}

export async function getModels(): Promise<{ models: ModelOption[]; default: string }> {
  return request("/models");
}

export async function getProperties(): Promise<{ properties: PropertyOption[] }> {
  return request("/properties");
}

export async function sendChat(params: {
  propertyCode: string;
  model: string;
  message: string;
  conversationId?: string;
}): Promise<ChatResponse> {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({
      property_code: params.propertyCode,
      model: params.model,
      message: params.message,
      conversation_id: params.conversationId
    })
  });
}

export async function approveAgentRun(params: {
  runId: string;
  propertyCode: string;
  conversationId?: string;
}): Promise<ChatResponse> {
  return request(`/api/agent-runs/${encodeURIComponent(params.runId)}/approve`, {
    method: "POST",
    body: JSON.stringify({
      property_code: params.propertyCode,
      conversation_id: params.conversationId,
      approved: true
    })
  });
}

function runScopeQuery(propertyCode: string, conversationId: string) {
  return new URLSearchParams({
    property_code: propertyCode,
    conversation_id: conversationId
  }).toString();
}

export async function getAgentRunTrace(params: {
  runId: string;
  propertyCode: string;
  conversationId: string;
}): Promise<AgentRunTrace> {
  const runPath = `/api/agent-runs/${encodeURIComponent(params.runId)}`;
  const query = runScopeQuery(params.propertyCode, params.conversationId);
  const [run, steps, events, citations] = await Promise.all([
    request<AgentRunDetail>(`${runPath}?${query}`),
    request<AgentRunStep[]>(`${runPath}/steps?${query}`),
    request<AgentRunEvent[]>(`${runPath}/events?${query}`),
    request<AgentRunCitation[]>(`${runPath}/citations?${query}`)
  ]);
  return { run, steps, events, citations };
}

export async function getAgentRunStatus(params: {
  runId: string;
  propertyCode: string;
  conversationId: string;
}): Promise<AgentRunDetail> {
  const query = runScopeQuery(params.propertyCode, params.conversationId);
  return request<AgentRunDetail>(
    `/api/agent-runs/${encodeURIComponent(params.runId)}?${query}`
  );
}

export async function cancelAgentRun(params: {
  runId: string;
  propertyCode: string;
  conversationId: string;
}): Promise<AgentRunDetail> {
  return request(`/api/agent-runs/${encodeURIComponent(params.runId)}/cancel`, {
    method: "POST",
    body: JSON.stringify({
      property_code: params.propertyCode,
      conversation_id: params.conversationId
    })
  });
}

type StreamHandlers = {
  onToken: (token: string) => void;
  onRunStarted?: (run: StreamStart) => void;
};

export type StreamStart = {
  run_id: string;
  conversation_id: string;
  property_code: string;
  reconnect_url: string;
};

function parseSseEvent(rawEvent: string): { event: string; data: string } | null {
  const lines = rawEvent.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));

  if (!eventLine || dataLines.length === 0) {
    return null;
  }

  return {
    event: eventLine.slice("event:".length).trim(),
    data: dataLines.map((line) => line.slice("data:".length).trimStart()).join("\n")
  };
}

export async function sendChatStream(
  params: {
    propertyCode: string;
    model: string;
    message: string;
    conversationId?: string;
    signal?: AbortSignal;
  },
  handlers: StreamHandlers
): Promise<ChatResponse> {
  const accessToken = await accessTokenProvider();
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {})
    },
    signal: params.signal,
    body: JSON.stringify({
      property_code: params.propertyCode,
      model: params.model,
      message: params.message,
      conversation_id: params.conversationId
    })
  });

  if (!response.ok || !response.body) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: ChatResponse | null = null;
  let startedRun: StreamStart | null = null;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const rawEvent of events) {
      const parsed = parseSseEvent(rawEvent);
      if (!parsed) {
        continue;
      }

      const payload = JSON.parse(parsed.data) as Record<string, unknown>;
      if (parsed.event === "token") {
        handlers.onToken(String(payload.delta ?? ""));
      } else if (parsed.event === "status" && payload.run_id) {
        startedRun = payload as StreamStart;
        handlers.onRunStarted?.(startedRun);
      } else if (parsed.event === "final") {
        finalResponse = payload as ChatResponse;
      } else if (parsed.event === "error") {
        throw new Error(String(payload.detail ?? "Streaming request failed."));
      }
    }

    if (done) {
      break;
    }
  }

  if (!finalResponse) {
    if (startedRun) {
      const status = await getAgentRunStatus({
        runId: startedRun.run_id,
        propertyCode: startedRun.property_code,
        conversationId: startedRun.conversation_id
      });
      throw new Error(
        `Streaming connection ended; run ${startedRun.run_id} is ${status.status}.`
      );
    }
    throw new Error("Streaming response ended before a run was created.");
  }

  return finalResponse;
}
