import { useEffect, useState } from "react";

import { ApiError, ScentAIClient } from "../lib/api-client";

export type RuntimeStatus = "idle" | "starting" | "ready" | "error";

function warmupError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "ScentAI servis bağlantısı doğrulanamadı.";
    if (error.status === 429) return "Başlatma limiti doldu. Bir dakika sonra tekrar dene.";
    return `ScentAI başlatılamadı: ${error.detail}`;
  }
  if (error instanceof TypeError) {
    return "ScentAI sunucusuna bağlanılamadı. Ağ bağlantısını ve API adresini kontrol et.";
  }
  if (error instanceof Error) return error.message;
  return "ScentAI başlatılırken beklenmeyen bir hata oluştu.";
}

export function useRuntimeWarmup(client: ScentAIClient | null, enabled: boolean) {
  const [status, setStatus] = useState<RuntimeStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!client || !enabled) {
      setStatus("idle");
      setError(null);
      return;
    }

    const controller = new AbortController();
    setStatus("starting");
    setError(null);
    void client.warmup(controller.signal).then(
      () => setStatus("ready"),
      (reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(warmupError(reason));
        setStatus("error");
      },
    );
    return () => controller.abort();
  }, [attempt, client, enabled]);

  return {
    status,
    error,
    retry: () => setAttempt((current) => current + 1),
  };
}
