import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, Leaf, MessageSquarePlus, RotateCcw, Settings, Square } from "lucide-react";
import ReactMarkdown from "react-markdown";

import "@fontsource/instrument-serif/latin-400.css";
import "@fontsource/instrument-serif/latin-ext-400.css";
import "@fontsource/geist/latin-400.css";
import "@fontsource/geist/latin-ext-400.css";
import "@fontsource/geist/latin-600.css";
import "@fontsource/geist/latin-ext-600.css";
import { ConnectionForm } from "../components/connection-form";
import { useChatSession, type ChatPendingPhase } from "../hooks/use-chat-session";
import { useRuntimeWarmup, type RuntimeStatus } from "../hooks/use-runtime-warmup";
import { ScentAIClient } from "../lib/api-client";
import { clearConnection, loadConnection, saveConnection } from "../lib/storage";
import type { ConnectionConfig } from "../types/api";
import type { ChatMessage } from "../types/chat";
import "./design-lab.css";

type LabView = "loading" | "empty" | "conversation" | "live";

const DIRECT_CONNECTION = import.meta.env.VITE_SCENTAI_DIRECT_CONNECTION === "true";
const DEFAULT_API_URL = import.meta.env.VITE_SCENTAI_API_URL ?? "/api/scentai";

const labViews: Array<{ id: LabView; label: string }> = [
  { id: "loading", label: "Yükleme" },
  { id: "empty", label: "Boş sohbet" },
  { id: "conversation", label: "Sohbet" },
  { id: "live", label: "Canlı" },
];

const demoMessages: ChatMessage[] = [
  {
    id: "demo-user",
    role: "user",
    content: "Bana vanilyalı, sıcak ama boğucu olmayan üç parfüm öner. Karakter farklarını da anlat.",
  },
  {
    id: "demo-assistant",
    role: "assistant",
    route: "llm_grounded",
    language: "tr",
    totalSeconds: 8.4,
    content: `Sıcaklığı yalnızca tatlılık üzerinden değil, bıraktığı izlenim üzerinden ayırdım.

**Tobacco Vanille**, tütün yaprağı ve kakao çevresinde daha koyu, törensel bir profil çiziyor. Soğuk bir akşamda oturmuş ve belirgin bir imza istiyorsan üçlüde en ciddi seçenek bu.

**Angels' Share**, tarçın ve pralin tarafıyla daha davetkar. Tatlılığı daha neşeli ve dışa dönük hissettirdiği için şık ama rahat bir akşam yemeğine kolayca uyum sağlar.

**Le Male Le Parfum** ise kakule, iris ve vanilyayı daha kontrollü taşıyor. Diğer ikisine göre daha temiz giyimli ve yakın mesafeli; dengeli bir başlangıç için benim seçimim bu olurdu.`,
    recommendations: [
      { perfume_id: 1, label: "Tobacco Vanille by Tom Ford", name: "Tobacco Vanille", brand: "Tom Ford" },
      { perfume_id: 2, label: "Angels' Share by By Kilian", name: "Angels' Share", brand: "By Kilian" },
      { perfume_id: 3, label: "Le Male Le Parfum by Jean Paul Gaultier", name: "Le Male Le Parfum", brand: "Jean Paul Gaultier" },
    ],
  },
];

function FlowerField({ settled = false }: { settled?: boolean }) {
  return (
    <div className={`flower-field ${settled ? "is-settled" : ""}`} aria-hidden="true">
      <img className="flower flower-iris" src="/design-lab/flowers/iris.png" alt="" />
      <img className="flower flower-rose" src="/design-lab/flowers/rose.png" alt="" />
      <img className="flower flower-jasmine" src="/design-lab/flowers/jasmine.png" alt="" />
      <img className="flower flower-poppy" src="/design-lab/flowers/poppy.png" alt="" />
      <div className="flower-pollen">
        {Array.from({ length: 16 }, (_, index) => <span key={index} />)}
      </div>
    </div>
  );
}

function BotanicalGate({
  status = "starting",
  error,
  onRetry,
}: {
  status?: RuntimeStatus;
  error?: string | null;
  onRetry?: () => void;
}) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (status !== "starting" && status !== "idle") return;
    const startedAt = Date.now();
    const timer = globalThis.setInterval(
      () => setElapsed(Math.floor((Date.now() - startedAt) / 1000)),
      1_000,
    );
    return () => globalThis.clearInterval(timer);
  }, [status]);

  const progress = status === "ready"
    ? 100
    : Math.min(94, Math.round(6 + 88 * (1 - Math.exp(-elapsed / 100))));
  const progressStage = status === "ready"
    ? "DANIŞMAN HAZIR"
    : progress < 28
      ? "ÇALIŞMA ORTAMI AYRILIYOR"
      : progress < 66
        ? "MODEL BELLEĞE ALINIYOR"
        : progress < 88
          ? "KOKU İNDEKSİ AÇILIYOR"
          : "DANIŞMAN HAZIRLANIYOR";

  return (
    <main className="botanical-gate">
      <FlowerField />
      <div className="gate-wash" aria-hidden="true" />
      <header className="gate-brand">
        <span>SCENTAI</span>
        <span>OLFACTORY COUNSEL · 2026</span>
      </header>
      <section className="gate-copy">
        <p className="lab-kicker">BOTANICAL INITIALISATION · {String(elapsed).padStart(2, "0")}</p>
        <h1>{status === "ready" ? <>Koku hafızası<br />hazır.</> : <>Koku hafızası<br />uyanıyor.</>}</h1>
        {status === "error" ? (
          <>
            <p className="gate-error" role="alert">{error}</p>
            <button className="gate-retry" type="button" onClick={onRetry}>
              <RotateCcw size={16} /> Yeniden dene
            </button>
          </>
        ) : (
          <p>Model, parfüm kataloğu ve botanik indeks hazırlanıyor. Çiçeklenme tamamlandığında konuşma kendiliğinden açılacak.</p>
        )}
        {status !== "error" ? (
          <div className="gate-inline-progress">
            <div className="gate-progress-meta" aria-hidden="true">
              <span>{progressStage}</span>
              <strong>%{String(progress).padStart(2, "0")}</strong>
            </div>
            <div
              className="gate-progress"
              role="progressbar"
              aria-label="ScentAI hazırlanıyor"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progress}
            >
              <span style={{ width: `${progress}%` }} />
            </div>
          </div>
        ) : null}
      </section>
      <p className="gate-footnote">PLEASE KEEP THIS WINDOW OPEN</p>
    </main>
  );
}

function PendingRecord({ phase }: { phase: ChatPendingPhase }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const timer = globalThis.setInterval(
      () => setElapsed(Math.floor((Date.now() - startedAt) / 1000)),
      1_000,
    );
    return () => globalThis.clearInterval(timer);
  }, []);

  const copy = phase === "queued"
    ? "İstek sıraya alındı"
    : phase === "running"
      ? "Danışman seçenekleri değerlendiriyor"
      : "İstek arşive iletiliyor";

  return (
    <article className="archive-message archive-assistant pending-record">
      <div className="message-index"><span>AI</span><small>{String(elapsed).padStart(2, "0")} SN</small></div>
      <div className="pending-bloom" aria-hidden="true">
        <span /><span /><span /><span />
      </div>
      <p>{copy}<span className="pending-dots">...</span></p>
    </article>
  );
}

function ArchiveComposer({
  submitting,
  onSend,
  onCancel,
}: {
  submitting: boolean;
  onSend: (query: string) => Promise<boolean>;
  onCancel: () => void;
}) {
  const [query, setQuery] = useState("");

  async function submit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const current = query.trim();
    if (!current || submitting) return;
    setQuery("");
    if (!(await onSend(current))) setQuery(current);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  return (
    <footer className="archive-composer-wrap">
      <form className="archive-composer" onSubmit={submit}>
        <label htmlFor="archive-query">DANIŞMANA SOR</label>
        <textarea
          id="archive-query"
          rows={1}
          maxLength={2000}
          value={query}
          disabled={submitting}
          placeholder="Bir his, ortam veya parfüm adıyla başla..."
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
        />
        {submitting ? (
          <button type="button" onClick={onCancel} title="İsteği iptal et">
            <Square size={18} /><span className="sr-only">İsteği iptal et</span>
          </button>
        ) : (
          <button type="submit" disabled={!query.trim()} title="Gönder">
            <ArrowUp size={20} /><span className="sr-only">Gönder</span>
          </button>
        )}
      </form>
    </footer>
  );
}

function RecommendationIndex({ message }: { message: Extract<ChatMessage, { role: "assistant" }> }) {
  if (!message.recommendations.length) return null;
  return (
    <ol className="scent-index" aria-label="Önerilen parfümler">
      {message.recommendations.map((candidate, index) => (
        <li key={candidate.perfume_id}>
          <span>{String(index + 1).padStart(2, "0")}</span>
          <strong>{candidate.name}</strong>
          <small>{candidate.brand}</small>
        </li>
      ))}
    </ol>
  );
}

function ArchiveThread({ messages, pendingPhase }: { messages: ChatMessage[]; pendingPhase: ChatPendingPhase | null }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pendingPhase]);

  if (!messages.length && !pendingPhase) {
    return (
      <section className="archive-empty">
        <p className="lab-kicker">NEW CONSULTATION · OPEN INDEX</p>
        <h1>Bugün nasıl<br />kokmak istersin?</h1>
        <p>Bir ortamı, hissi veya sevdiğin bir parfümü anlat. İstersen iki kokuyu karşılaştır, istemediğin notaları dışarıda bırak.</p>
        <div className="empty-prompts" aria-hidden="true">
          <span>01 · Temiz bir ofis kokusu</span>
          <span>02 · Daha az dumanlı bir Aventus alternatifi</span>
          <span>03 · Baharatlı bir date parfümü</span>
        </div>
      </section>
    );
  }

  let assistantNumber = 0;
  return (
    <section className="archive-thread" aria-label="Konuşma" aria-live="polite">
      {messages.map((message, index) => {
        if (message.role === "user") {
          return (
            <article className="archive-message archive-user" key={message.id}>
              <div className="message-index"><span>Q{String(index + 1).padStart(2, "0")}</span><small>SEN</small></div>
              <p>{message.content}</p>
            </article>
          );
        }
        assistantNumber += 1;
        return (
          <article className="archive-message archive-assistant" key={message.id}>
            <div className="message-index">
              <span>A{String(assistantNumber).padStart(2, "0")}</span>
              <small>{message.totalSeconds.toFixed(1)} SN</small>
            </div>
            <div className="assistant-body">
              <div className="assistant-meta">
                <span>SCENTAI · CURATED RESPONSE</span>
                <span>{message.language.toUpperCase()} · {message.route.replaceAll("_", " ")}</span>
              </div>
              <div className="assistant-copy"><ReactMarkdown>{message.content}</ReactMarkdown></div>
              <RecommendationIndex message={message} />
            </div>
          </article>
        );
      })}
      {pendingPhase ? <PendingRecord phase={pendingPhase} /> : null}
      <div ref={endRef} />
    </section>
  );
}

function ArchiveHeader({
  onNewConversation,
  onSettings,
}: {
  onNewConversation: () => void;
  onSettings?: () => void;
}) {
  return (
    <header className="archive-header">
      <a href="?preview=design&screen=live" className="archive-brand">
        <span className="archive-brand-seal"><Leaf size={15} /></span>
        <span><strong>SCENTAI</strong><small>OLFACTORY COUNSEL</small></span>
      </a>
      <div className="archive-status"><i /> CONSULTANT READY</div>
      <nav aria-label="Konuşma araçları">
        {onSettings ? <button type="button" onClick={onSettings} title="Bağlantı"><Settings size={17} /></button> : null}
        <button type="button" onClick={onNewConversation} title="Yeni konuşma">
          <MessageSquarePlus size={17} /><span>YENİ KONUŞMA</span>
        </button>
      </nav>
    </header>
  );
}

function ArchiveChat({
  messages,
  pendingPhase,
  submitting,
  error,
  onSend,
  onCancel,
  onNewConversation,
  onClearError,
  onSettings,
}: {
  messages: ChatMessage[];
  pendingPhase: ChatPendingPhase | null;
  submitting: boolean;
  error: string | null;
  onSend: (query: string) => Promise<boolean>;
  onCancel: () => void;
  onNewConversation: () => void;
  onClearError: () => void;
  onSettings?: () => void;
}) {
  return (
    <div className="archive-app">
      <ArchiveHeader onNewConversation={onNewConversation} onSettings={onSettings} />
      <FlowerField settled />
      <aside className="archive-rail" aria-hidden="true">
        <span>ACTIVE CONSULTATION</span>
        <span>131,930 CATALOGUE RECORDS</span>
        <span>GROUNDED RESPONSE</span>
      </aside>
      <main className="archive-main">
        <ArchiveThread messages={messages} pendingPhase={pendingPhase} />
      </main>
      {error ? (
        <div className="archive-error" role="alert">
          <span>{error}</span><button type="button" onClick={onClearError}>Kapat</button>
        </div>
      ) : null}
      <ArchiveComposer submitting={submitting} onSend={onSend} onCancel={onCancel} />
    </div>
  );
}

function DemoExperience({ empty = false }: { empty?: boolean }) {
  const [messages, setMessages] = useState<ChatMessage[]>(empty ? [] : demoMessages);
  const [pending, setPending] = useState<ChatPendingPhase | null>(null);

  async function demoSend(query: string) {
    setMessages((current) => [...current, { id: `demo-${Date.now()}`, role: "user", content: query }]);
    setPending("running");
    globalThis.setTimeout(() => setPending(null), 2_400);
    return true;
  }

  return (
    <ArchiveChat
      messages={messages}
      pendingPhase={pending}
      submitting={Boolean(pending)}
      error={null}
      onSend={demoSend}
      onCancel={() => setPending(null)}
      onNewConversation={() => setMessages([])}
      onClearError={() => undefined}
    />
  );
}

function LiveExperience() {
  const [connection, setConnection] = useState<ConnectionConfig>(() =>
    DIRECT_CONNECTION ? loadConnection(DEFAULT_API_URL) : { apiUrl: DEFAULT_API_URL, apiKey: "" },
  );
  const [showConnection, setShowConnection] = useState(DIRECT_CONNECTION && !connection.apiKey);
  const [readyTransitionComplete, setReadyTransitionComplete] = useState(false);
  const client = useMemo(() => {
    if (DIRECT_CONNECTION && !connection.apiKey) return null;
    try { return new ScentAIClient(connection); } catch { return null; }
  }, [connection]);
  const runtime = useRuntimeWarmup(client, Boolean(client) && !showConnection);
  const chat = useChatSession(connection.apiUrl, client);

  useEffect(() => {
    if (runtime.status !== "ready") {
      setReadyTransitionComplete(false);
      return;
    }
    const timer = globalThis.setTimeout(() => setReadyTransitionComplete(true), 850);
    return () => globalThis.clearTimeout(timer);
  }, [runtime.status]);

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

  if ((DIRECT_CONNECTION && showConnection) || !client) {
    return <ConnectionForm initialValue={connection} onSubmit={connect} />;
  }
  if (runtime.status !== "ready" || !readyTransitionComplete) {
    return <BotanicalGate status={runtime.status} error={runtime.error} onRetry={runtime.retry} />;
  }

  return (
    <ArchiveChat
      messages={chat.messages}
      pendingPhase={chat.pendingPhase}
      submitting={chat.status === "submitting"}
      error={chat.error}
      onSend={chat.send}
      onCancel={chat.cancel}
      onNewConversation={() => void chat.newConversation()}
      onClearError={chat.clearError}
      onSettings={DIRECT_CONNECTION ? () => setShowConnection(true) : undefined}
    />
  );
}

function LabSwitcher({ view, onChange }: { view: LabView; onChange: (view: LabView) => void }) {
  return (
    <div className="lab-switcher" aria-label="Tasarım laboratuvarı görünümü">
      {labViews.map((item) => (
        <button key={item.id} type="button" className={item.id === view ? "active" : ""} onClick={() => onChange(item.id)}>
          {item.label}
        </button>
      ))}
    </div>
  );
}

export default function DesignLab() {
  const [view, setView] = useState<LabView>(() => {
    const requested = new URLSearchParams(window.location.search).get("screen");
    return labViews.some((item) => item.id === requested) ? requested as LabView : "conversation";
  });

  useEffect(() => {
    document.body.classList.add("design-lab-body");
    window.scrollTo(0, window.scrollY);
    return () => document.body.classList.remove("design-lab-body");
  }, []);

  function changeView(next: LabView) {
    setView(next);
    const url = new URL(window.location.href);
    url.searchParams.set("screen", next);
    window.history.replaceState({}, "", url);
  }

  return (
    <div className="design-lab">
      {view === "loading" ? <BotanicalGate /> : null}
      {view === "empty" ? <DemoExperience empty /> : null}
      {view === "conversation" ? <DemoExperience /> : null}
      {view === "live" ? <LiveExperience /> : null}
      <LabSwitcher view={view} onChange={changeView} />
    </div>
  );
}
