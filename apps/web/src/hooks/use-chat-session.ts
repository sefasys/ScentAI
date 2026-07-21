import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, ScentAIClient, type JobProgress } from "../lib/api-client";
import { clearConversation, loadConversation, saveConversation } from "../lib/storage";
import type { ChatMessage } from "../types/chat";

export type ChatStatus = "idle" | "submitting";
export type ChatPendingPhase = "submitting" | JobProgress;

function messageId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "ScentAI servis bağlantısı doğrulanamadı.";
    if (error.status === 404) return "İstek veya konuşma oturumu artık bulunamıyor.";
    if (error.status === 408) return "Yanıt beklenenden uzun sürdü. Mesajını yeniden gönderebilirsin.";
    if (error.status === 429) return "Çok sayıda istek gönderildi. Bir dakika sonra tekrar dene.";
    if ([502, 503, 504].includes(error.status)) {
      return "ScentAI servisi geçici olarak yeniden başlatılıyor. Biraz sonra tekrar dene.";
    }
    return `ScentAI isteği tamamlanamadı: ${error.detail}`;
  }
  if (error instanceof TypeError) {
    return "ScentAI sunucusuna bağlanılamadı. Ağ bağlantısını ve API adresini kontrol et.";
  }
  if (error instanceof Error) return error.message;
  return "Beklenmeyen bir hata oluştu.";
}

export function useChatSession(apiUrl: string, client: ScentAIClient | null) {
  const restored = loadConversation(apiUrl);
  const [messages, setMessages] = useState<ChatMessage[]>(restored.messages);
  const [sessionId, setSessionId] = useState<string | null>(restored.sessionId);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [pendingPhase, setPendingPhase] = useState<ChatPendingPhase | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const submittingRef = useRef(false);
  const currentApiUrl = useRef(apiUrl);

  useEffect(() => {
    if (currentApiUrl.current === apiUrl) return;
    currentApiUrl.current = apiUrl;
    const next = loadConversation(apiUrl);
    setMessages(next.messages);
    setSessionId(next.sessionId);
    setError(null);
  }, [apiUrl]);

  useEffect(() => {
    saveConversation(apiUrl, sessionId, messages);
  }, [apiUrl, messages, sessionId]);

  const send = useCallback(
    async (query: string) => {
      const cleaned = query.trim();
      if (!client || !cleaned || submittingRef.current) return false;

      submittingRef.current = true;
      setError(null);
      setStatus("submitting");
      setPendingPhase("submitting");
      const userMessage = { id: messageId(), role: "user" as const, content: cleaned };
      setMessages((current) => [...current, userMessage]);
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        let response: Awaited<ReturnType<ScentAIClient["chat"]>>;
        try {
          response = await client.chat(
            {
              query: cleaned,
              session_id: sessionId ?? undefined,
            },
            controller.signal,
            setPendingPhase,
          );
        } catch (requestError) {
          if (!(requestError instanceof ApiError && requestError.status === 404 && sessionId)) {
            throw requestError;
          }
          // Modal scale-down clears in-memory sessions. Replaying once without the stale
          // session keeps the user's current request intact after a cold restart.
          setSessionId(null);
          response = await client.chat({ query: cleaned }, controller.signal, setPendingPhase);
        }
        setSessionId(response.session_id);
        setMessages((current) => [
          ...current,
          {
            id: response.request_id,
            role: "assistant",
            content: response.answer,
            recommendations: response.recommendations,
            route: response.route,
            language: response.language,
            totalSeconds: response.total_seconds,
          },
        ]);
        return true;
      } catch (requestError) {
        setMessages((current) => current.filter((message) => message.id !== userMessage.id));
        if (requestError instanceof DOMException && requestError.name === "AbortError") {
          setError("İstek iptal edildi.");
        } else {
          if (requestError instanceof ApiError && requestError.status === 404) setSessionId(null);
          setError(errorMessage(requestError));
        }
        return false;
      } finally {
        abortRef.current = null;
        submittingRef.current = false;
        setStatus("idle");
        setPendingPhase(null);
      }
    },
    [client, sessionId],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    submittingRef.current = false;
  }, []);

  const newConversation = useCallback(async () => {
    abortRef.current?.abort();
    const previousSessionId = sessionId;
    setMessages([]);
    setSessionId(null);
    setError(null);
    clearConversation();
    if (client && previousSessionId) {
      try {
        await client.deleteSession(previousSessionId);
      } catch (deleteError) {
        setError(`Eski sunucu oturumu silinemedi: ${errorMessage(deleteError)}`);
      }
    }
  }, [client, sessionId]);

  return {
    messages,
    sessionId,
    status,
    pendingPhase,
    error,
    send,
    cancel,
    newConversation,
    clearError: () => setError(null),
  };
}
