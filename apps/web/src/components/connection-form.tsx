import { useState, type FormEvent } from "react";

import type { ConnectionConfig } from "../types/api";

interface ConnectionFormProps {
  initialValue: ConnectionConfig;
  onSubmit: (config: ConnectionConfig) => string | null;
}

export function ConnectionForm({ initialValue, onSubmit }: ConnectionFormProps) {
  const [apiUrl, setApiUrl] = useState(initialValue.apiUrl);
  const [apiKey, setApiKey] = useState(initialValue.apiKey);
  const [error, setError] = useState<string | null>(null);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const submitError = onSubmit({ apiUrl: apiUrl.trim(), apiKey: apiKey.trim() });
    setError(submitError);
  }

  return (
    <main className="connection-page">
      <form className="connection-form" onSubmit={submit}>
        <header>
          <p className="product-name">ScentAI</p>
          <h1>API bağlantısı</h1>
        </header>

        <label htmlFor="api-url">API adresi</label>
        <input
          id="api-url"
          type="url"
          required
          autoComplete="url"
          value={apiUrl}
          onChange={(event) => setApiUrl(event.target.value)}
        />

        <label htmlFor="api-key">API key</label>
        <input
          id="api-key"
          type="password"
          required
          autoComplete="off"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
        />

        {error ? <p className="form-error" role="alert">{error}</p> : null}
        <button className="primary-button" type="submit">Bağlan</button>
      </form>
    </main>
  );
}
