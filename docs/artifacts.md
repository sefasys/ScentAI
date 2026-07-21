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
- The raw 131,930-record perfume catalog is not republished by this repository.
- Model adapters may have terms inherited from the base model and training inputs.
- Deployment secrets must never be stored beside source code.

## Dataset Distribution

The 32,000-record synthetic conversation corpus is a derivative of [Le Decanteur's Fragrantica
Perfumes dataset](https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes), which is
licensed under CC BY-NC-SA 4.0. It is prepared for separate Kaggle distribution under the same
license. The Git repository contains its dataset card, attribution, statistics, sample, and
reproducible packaging tools, while the compressed JSONL exports remain in the dataset registry.

The Kaggle package deliberately contains generated conversations rather than a second copy of the
source catalog. See [`dataset/README.md`](../dataset/README.md) for intended uses, limitations, and
the relationship between the Gemma and OpenAI-compatible exports.

## Model Distribution

Model adapters should live in a dedicated model registry only after their inherited base-model and
training-data terms have been reviewed. Chroma and SQLite runtime snapshots remain deployment
artifacts rather than public dataset files.

Historical release manifests are retained under [`evaluation/historical_manifests`](../evaluation/historical_manifests/) for provenance. Their paths describe the original development workspace and are not current installation instructions.
