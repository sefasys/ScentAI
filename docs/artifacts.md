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

The [ScentAI 32K Grounded Perfume Conversations](https://www.kaggle.com/datasets/sefasoysal/scentai-32k-grounded-perfume-conversations)
corpus is published on Kaggle. It is a derivative of [Le Decanteur's Fragrantica
Perfumes dataset](https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes), which is
licensed under CC BY-NC-SA 4.0, and is distributed under the same license. The Git repository
contains its dataset card, attribution, statistics, sample, and
reproducible packaging tools, while the compressed JSONL exports remain in the dataset registry.

The Kaggle package deliberately contains generated conversations rather than a second copy of the
source catalog. See [`dataset/README.md`](../dataset/README.md) for intended uses, limitations, and
the relationship between the Gemma and OpenAI-compatible exports.

## Model Distribution

The evaluated adapter is published in the dedicated
[ScentAI Gemma 4 12B LoRA repository](https://huggingface.co/sefasys/scentai-gemma-4-12b-it-lora)
after review of the inherited base-model and training-data terms. It contains only
`adapter_config.json`, `adapter_model.safetensors`, its model card, license, and release manifest;
the Gemma 4 base weights remain separately distributed by Google.

The adapter files can be downloaded through PEFT/Transformers or with `snapshot_download` from
`huggingface_hub`. Chroma and SQLite runtime snapshots remain deployment artifacts rather than
public dataset files because they contain a repackaged index/catalog representation and are not
required to inspect or import the adapter itself.

Historical release manifests are retained under [`evaluation/historical_manifests`](../evaluation/historical_manifests/) for provenance. Their paths describe the original development workspace and are not current installation instructions.
