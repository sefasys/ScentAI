import type { ConnectionConfig } from "../types/api";
import type { ChatMessage, StoredConversation } from "../types/chat";

const CONNECTION_KEY = "scentai.connection.v1";
const CONVERSATION_KEY = "scentai.conversation.v1";
const MAX_STORED_MESSAGES = 100;

function sessionStore(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function persistentStore(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function validMessages(value: unknown): ChatMessage[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((message): message is ChatMessage => {
      if (!message || typeof message !== "object") return false;
      const candidate = message as Partial<ChatMessage>;
      return (
        typeof candidate.id === "string" &&
        typeof candidate.content === "string" &&
        (candidate.role === "user" ||
          (candidate.role === "assistant" &&
            Array.isArray(candidate.recommendations) &&
            typeof candidate.route === "string" &&
            typeof candidate.language === "string" &&
            typeof candidate.totalSeconds === "number"))
      );
    })
    .slice(-MAX_STORED_MESSAGES);
}

export function loadConnection(defaultApiUrl: string): ConnectionConfig {
  const fallback = { apiUrl: defaultApiUrl, apiKey: "" };
  const raw = sessionStore()?.getItem(CONNECTION_KEY);
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as Partial<ConnectionConfig>;
    return {
      apiUrl: typeof parsed.apiUrl === "string" ? parsed.apiUrl : defaultApiUrl,
      apiKey: typeof parsed.apiKey === "string" ? parsed.apiKey : "",
    };
  } catch {
    return fallback;
  }
}

export function saveConnection(config: ConnectionConfig): void {
  sessionStore()?.setItem(CONNECTION_KEY, JSON.stringify(config));
}

export function clearConnection(): void {
  sessionStore()?.removeItem(CONNECTION_KEY);
}

export function loadConversation(apiUrl: string): Pick<StoredConversation, "sessionId" | "messages"> {
  const empty = { sessionId: null, messages: [] as ChatMessage[] };
  const local = persistentStore();
  const legacySession = sessionStore();
  const raw = local?.getItem(CONVERSATION_KEY) ?? legacySession?.getItem(CONVERSATION_KEY);
  if (!raw) return empty;
  try {
    const parsed = JSON.parse(raw) as Partial<StoredConversation> & { version?: number };
    if (![1, 2].includes(parsed.version ?? 0) || parsed.apiUrl !== apiUrl) {
      return empty;
    }
    const messages = validMessages(parsed.messages);
    if (legacySession?.getItem(CONVERSATION_KEY) && !local?.getItem(CONVERSATION_KEY)) {
      saveConversation(
        apiUrl,
        typeof parsed.sessionId === "string" ? parsed.sessionId : null,
        messages,
      );
      legacySession.removeItem(CONVERSATION_KEY);
    }
    return {
      sessionId: typeof parsed.sessionId === "string" ? parsed.sessionId : null,
      messages,
    };
  } catch {
    return empty;
  }
}

export function saveConversation(
  apiUrl: string,
  sessionId: string | null,
  messages: ChatMessage[],
): void {
  const value: StoredConversation = {
    version: 2,
    apiUrl,
    sessionId,
    messages: messages.slice(-MAX_STORED_MESSAGES),
    updatedAt: new Date().toISOString(),
  };
  try {
    persistentStore()?.setItem(CONVERSATION_KEY, JSON.stringify(value));
  } catch {
    // Chat remains usable when storage is unavailable or its browser quota is full.
  }
}

export function clearConversation(): void {
  persistentStore()?.removeItem(CONVERSATION_KEY);
  sessionStore()?.removeItem(CONVERSATION_KEY);
}
