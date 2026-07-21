# Artifacts And Data

The public source repository is deliberately artifact-free.

## Required Runtime Layout

The deployment tools expect compatible equivalents of:

```text
models/
└── scentai/
    ├── adapter_config.json
    └── adapter_model.safetensors

chroma_db_bge_m3/
└── chroma.sqlite3

scentai_catalog.sqlite3
```

The evaluated catalog contained 131,930 perfumes and 692,729 directed similarity edges. The Chroma collection used the same perfume count and BAAI/bge-m3 embeddings.

## Why They Are Excluded

- Git is not suitable for model checkpoints or vector snapshots of this size.
- Raw catalog redistribution requires a separate source-license review.
- Model adapters may have terms inherited from the base model and training inputs.
- Deployment secrets must never be stored beside source code.

## Distribution Plan

When distribution rights are established, model adapters should live in a dedicated model registry and datasets in a dedicated dataset registry. GitHub should contain checksums, schemas, small samples, and build scripts rather than the binary artifacts themselves.

Historical release manifests are retained under [`evaluation/historical_manifests`](../evaluation/historical_manifests/) for provenance. Their paths describe the original development workspace and are not current installation instructions.

