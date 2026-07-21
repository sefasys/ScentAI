import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";

interface ComposerProps {
  submitting: boolean;
  onSend: (query: string) => Promise<boolean>;
  onCancel: () => void;
}

export function Composer({ submitting, onSend, onCancel }: ComposerProps) {
  const [query, setQuery] = useState("");

  async function submit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (!query.trim() || submitting) return;
    const currentQuery = query;
    setQuery("");
    const sent = await onSend(currentQuery);
    if (!sent) setQuery(currentQuery);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label className="sr-only" htmlFor="chat-query">Mesaj</label>
      <textarea
        id="chat-query"
        rows={2}
        maxLength={2000}
        placeholder="Parfüm tercihini yaz..."
        value={query}
        disabled={submitting}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      {submitting ? (
        <button className="icon-button" type="button" onClick={onCancel} title="İsteği iptal et">
          <Square size={18} aria-hidden="true" />
          <span className="sr-only">İsteği iptal et</span>
        </button>
      ) : (
        <button className="icon-button" type="submit" disabled={!query.trim()} title="Gönder">
          <Send size={19} aria-hidden="true" />
          <span className="sr-only">Gönder</span>
        </button>
      )}
    </form>
  );
}
