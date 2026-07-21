/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SCENTAI_API_URL?: string;
  readonly VITE_SCENTAI_DIRECT_CONNECTION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
