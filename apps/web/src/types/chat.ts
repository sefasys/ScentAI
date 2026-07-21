import type { CandidateSummary } from "./api";

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  recommendations: CandidateSummary[];
  route: string;
  language: string;
  totalSeconds: number;
}

export type ChatMessage = UserMessage | AssistantMessage;

export interface StoredConversation {
  version: 2;
  apiUrl: string;
  sessionId: string | null;
  messages: ChatMessage[];
  updatedAt: string;
}
