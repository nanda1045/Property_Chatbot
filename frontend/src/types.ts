export type ModelOption = {
  id: string;
  label: string;
  provider: string;
};

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
  answer_markdown: string;
  components: UIComponent[];
  sources: Source[];
  tool_results: Record<string, unknown>;
};

export type ChatTurn = {
  id: string;
  question: string;
  streamedAnswer?: string;
  response?: ChatResponse;
  error?: string;
};
