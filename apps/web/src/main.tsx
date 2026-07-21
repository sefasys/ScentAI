import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles.css";

const designPreview = new URLSearchParams(window.location.search).get("preview") === "design";
const DesignLab = lazy(() => import("./design-lab/DesignLab"));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {designPreview ? (
      <Suspense fallback={null}><DesignLab /></Suspense>
    ) : <App />}
  </StrictMode>,
);
