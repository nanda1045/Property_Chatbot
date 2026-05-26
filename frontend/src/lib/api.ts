import type { ChatResponse, ModelOption, PropertyOption } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
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
}): Promise<ChatResponse> {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({
      property_code: params.propertyCode,
      model: params.model,
      message: params.message
    })
  });
}

type StreamHandlers = {
  onToken: (token: string) => void;
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
  },
  handlers: StreamHandlers
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      property_code: params.propertyCode,
      model: params.model,
      message: params.message
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
    throw new Error("Streaming response ended before the final answer was received.");
  }

  return finalResponse;
}
