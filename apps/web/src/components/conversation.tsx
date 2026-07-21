import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import type { ChatMessage } from "../types/chat";
import type { ChatPendingPhase } from "../hooks/use-chat-session";

interface ConversationProps {
  messages: ChatMessage[];
  pendingPhase: ChatPendingPhase | null;
}

function PendingMessage({ phase }: { phase: ChatPendingPhase }) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const timer = globalThis.setInterval(
      () => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)),
      1_000,
    );
    return () => globalThis.clearInterval(timer);
  }, []);

  const label =
    phase === "queued"
      ? "İstek sıraya alındı..."
      : phase === "running"
        ? "Parfüm danışmanı yanıtı hazırlıyor..."
        : "İstek gönderiliyor...";

  return (
    <article className="message message-assistant pending-message">
      <p className="message-role">ScentAI</p>
      <p>{label}</p>
      {elapsedSeconds >= 20 ? (
        <p className="pending-detail">İşlem devam ediyor · {elapsedSeconds} sn</p>
      ) : null}
    </article>
  );
}

export function Conversation({ messages, pendingPhase }: ConversationProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages, pendingPhase]);

  if (!messages.length && !pendingPhase) {
    return (
      <section className="empty-conversation" aria-label="Yeni konuşma">
        <h1>Nasıl bir parfüm arıyorsun?</h1>
        <p>Kullanım alanını, sevdiğin karakteri veya karşılaştırmak istediğin parfümleri yaz.</p>
      </section>
    );
  }

  return (
    <section className="conversation" aria-label="Konuşma" aria-live="polite">
      {messages.map((message) => (
        <article className={`message message-${message.role}`} key={message.id}>
          <p className="message-role">{message.role === "user" ? "Sen" : "ScentAI"}</p>
          {message.role === "assistant" ? (
            <>
              <div className="answer"><ReactMarkdown>{message.content}</ReactMarkdown></div>
              {message.recommendations.length ? (
                <ul className="recommendations" aria-label="Önerilen parfümler">
                  {message.recommendations.map((candidate) => (
                    <li key={candidate.perfume_id}>
                      <strong>{candidate.name}</strong>
                      <span>{candidate.brand}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
              <p className="message-meta">{message.totalSeconds.toFixed(1)} sn</p>
            </>
          ) : (
            <p>{message.content}</p>
          )}
        </article>
      ))}
      {pendingPhase ? <PendingMessage phase={pendingPhase} /> : null}
      <div ref={endRef} />
    </section>
  );
}
