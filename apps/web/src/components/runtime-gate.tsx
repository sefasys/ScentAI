import { useEffect, useState } from "react";
import { RefreshCw, Settings } from "lucide-react";

interface RuntimeGateProps {
  error: string | null;
  onRetry: () => void;
  onSettings?: () => void;
}

export function RuntimeGate({ error, onRetry, onSettings }: RuntimeGateProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (error) return;
    const startedAt = Date.now();
    const timer = window.setInterval(
      () => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [error]);

  const progressText =
    elapsedSeconds < 30
      ? "Güvenli sunucu bağlantısı kuruluyor."
      : elapsedSeconds < 180
        ? "GPU çalışma ortamı başlatılıyor."
        : "Model ve parfüm kataloğu belleğe yükleniyor.";

  const elapsedText =
    elapsedSeconds < 60
      ? `${elapsedSeconds} sn`
      : `${Math.floor(elapsedSeconds / 60)} dk ${elapsedSeconds % 60} sn`;

  return (
    <main className="runtime-gate" aria-live="polite">
      <div className="runtime-gate-content">
        <p className="runtime-eyebrow">ScentAI</p>
        <h1>{error ? "Başlatma tamamlanamadı" : "Parfüm danışmanı hazırlanıyor"}</h1>
        {error ? (
          <p className="runtime-error" role="alert">{error}</p>
        ) : (
          <>
            <p>{progressText} İlk açılış birkaç dakika sürebilir; hazır olduğunda sohbet otomatik olarak açılacak.</p>
            <div className="runtime-progress" aria-label="Model yükleniyor"><span /></div>
            <p className="runtime-elapsed">Geçen süre: {elapsedText} · Bu sayfayı açık bırakabilirsin.</p>
          </>
        )}
        <div className="runtime-actions">
          {error ? (
            <button type="button" onClick={onRetry}>
              <RefreshCw size={17} aria-hidden="true" />
              Tekrar dene
            </button>
          ) : null}
          {onSettings ? (
            <button type="button" onClick={onSettings}>
              <Settings size={17} aria-hidden="true" />
              Bağlantı ayarları
            </button>
          ) : null}
        </div>
      </div>
    </main>
  );
}
