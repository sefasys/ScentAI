import { useMemo, useState } from "react";
import { LogOut, MessageSquarePlus, Settings } from "lucide-react";

import { Composer } from "./components/composer";
import { ConnectionForm } from "./components/connection-form";
import { Conversation } from "./components/conversation";
import { RuntimeGate } from "./components/runtime-gate";
import { useChatSession } from "./hooks/use-chat-session";
import { useRuntimeWarmup } from "./hooks/use-runtime-warmup";
import { ScentAIClient } from "./lib/api-client";
import { clearConnection, loadConnection, saveConnection } from "./lib/storage";
import type { ConnectionConfig } from "./types/api";

const DIRECT_CONNECTION = import.meta.env.VITE_SCENTAI_DIRECT_CONNECTION === "true";
const DEFAULT_API_URL = import.meta.env.VITE_SCENTAI_API_URL ?? "/api/scentai";

export default function App() {
  const [connection, setConnection] = useState<ConnectionConfig>(() =>
    DIRECT_CONNECTION
      ? loadConnection(DEFAULT_API_URL)
      : { apiUrl: DEFAULT_API_URL, apiKey: "" },
  );
  const [showConnection, setShowConnection] = useState(
    DIRECT_CONNECTION && !connection.apiKey,
  );
  const client = useMemo(() => {
    if (DIRECT_CONNECTION && !connection.apiKey) return null;
    try {
      return new ScentAIClient(connection);
    } catch {
      return null;
    }
  }, [connection]);
  const runtime = useRuntimeWarmup(client, Boolean(client) && !showConnection);
  const chat = useChatSession(connection.apiUrl, client);

  function connect(config: ConnectionConfig): string | null {
    try {
      new ScentAIClient(config);
      saveConnection(config);
      setConnection(config);
      setShowConnection(false);
      return null;
    } catch (error) {
      return error instanceof Error ? error.message : "Bağlantı bilgileri geçersiz.";
    }
  }

  function disconnect() {
    clearConnection();
    setConnection((current) => ({ ...current, apiKey: "" }));
    setShowConnection(true);
  }

  if ((DIRECT_CONNECTION && showConnection) || !client) {
    return <ConnectionForm initialValue={connection} onSubmit={connect} />;
  }

  if (runtime.status !== "ready") {
    return (
      <RuntimeGate
        error={runtime.error}
        onRetry={runtime.retry}
        onSettings={DIRECT_CONNECTION ? () => setShowConnection(true) : undefined}
      />
    );
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <strong>ScentAI</strong>
          <span className="connection-status">Hazır</span>
        </div>
        <nav aria-label="Konuşma araçları">
          <button type="button" onClick={() => void chat.newConversation()}>
            <MessageSquarePlus size={17} aria-hidden="true" />
            Yeni konuşma
          </button>
          {DIRECT_CONNECTION ? (
            <>
              <button type="button" onClick={() => setShowConnection(true)}>
                <Settings size={17} aria-hidden="true" />
                Bağlantı
              </button>
              <button type="button" onClick={disconnect} title="Bağlantıyı kes">
                <LogOut size={17} aria-hidden="true" />
                <span className="sr-only">Bağlantıyı kes</span>
              </button>
            </>
          ) : null}
        </nav>
      </header>

      <Conversation messages={chat.messages} pendingPhase={chat.pendingPhase} />

      <footer className="composer-area">
        {chat.error ? (
          <div className="request-error" role="alert">
            <span>{chat.error}</span>
            <button type="button" onClick={chat.clearError}>Kapat</button>
          </div>
        ) : null}
        <Composer
          submitting={chat.status === "submitting"}
          onSend={chat.send}
          onCancel={chat.cancel}
        />
      </footer>
    </div>
  );
}
