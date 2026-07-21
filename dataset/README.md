# ScentAI 32K Grounded Perfume Conversations

ScentAI 32K is a synthetic instruction-tuning dataset for grounded perfume consultation. It contains
32,000 three-message conversations built from structured perfume cards and organized as a five-level
curriculum: factual discipline, filtering, semantic recommendation, grounded explanation, and
preference-aware consultation.

This dataset is the training corpus used by the ScentAI research project. It is not a dump of the
underlying perfume catalog and contains no real user conversations.

## Files

| File | Records | Assistant role | Purpose |
| --- | ---: | --- | --- |
| `scentai_train_gemma.jsonl.gz` | 30,400 | `model` | Stratified training split |
| `scentai_validation_gemma.jsonl.gz` | 1,600 | `model` | Stratified validation split |
| `scentai_full_openai.jsonl.gz` | 32,000 | `assistant` | Generic/OpenAI-style full export |
| `statistics.json` | - | - | Reproducible corpus statistics |
| `manifest.json` | - | - | File sizes, hashes, roles, and record counts |

The Gemma train and validation files together contain the same 32,000 examples as the OpenAI-style
file. Do not concatenate all three files for training; doing so would duplicate the corpus.

## Record Schema

Each line is one JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "model", "content": "..."}
  ]
}
```

The OpenAI-style export changes only the final role from `model` to `assistant`. Debug metadata used
during generation is not included in the public exports.

## Curriculum

| Level | Records | Share | Primary behavior |
| --- | ---: | ---: | --- |
| L1 | 4,000 | 12.50% | Factual perfume knowledge and concise answers |
| L2 | 6,000 | 18.75% | Strict filtering, ranking, comparison, and no-match behavior |
| L3 | 7,000 | 21.88% | Natural semantic recommendation requests and list responses |
| L4 | 10,000 | 31.25% | Grounded recommendations with short explanations |
| L5 | 5,000 | 15.62% | Synthetic preference profiles and personalization |

The 95/5 train-validation split is stratified by level:

| Level | Train | Validation |
| --- | ---: | ---: |
| L1 | 3,800 | 200 |
| L2 | 5,700 | 300 |
| L3 | 6,650 | 350 |
| L4 | 9,500 | 500 |
| L5 | 4,750 | 250 |

Split and shuffle seed: `20260629`.

## Construction

1. Structured perfume records were normalized into evidence cards.
2. L1 examples were produced from factual templates.
3. L2 examples were generated combinatorially and checked by deterministic validators.
4. L3-L5 natural-language query wording was generated through an OpenRouter provider pool using
   Llama 3.3 70B Instruct and GPT-4.1 mini after direct and weighted preflight tests.
5. Category-specific generators selected only evidence-supported answers from the supplied cards.
6. Generation used `fallback_policy=fail`; failed API generations were not silently replaced by
   hard-coded query text.
7. Chunk-level duplicate and quality checks were run before the final seeded merge.
8. Debug metadata was removed and two role-compatible exports were produced.

The corpus is primarily English. L5 profiles are synthetic and do not represent real individuals.

The final audit found 81 repeated full-record instances out of 32,000 records (0.253%): 79 in L2,
one in L1, and one in L3. These are retained because this release reproduces the corpus used for
the reported model training. Consumers who require a strictly unique corpus can deduplicate by the
ordered `(system, user, answer)` tuple. The audit also reports repeated user prompts separately,
because the same wording paired with different evidence or answers is not necessarily a duplicate.

## Source Data And Attribution

Perfume facts used to construct the evidence cards derive from **Fragrantica Perfumes: Ratings,
Notes, Votes & More**, published by Le Decanteur on Kaggle:

- Source: https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes
- Source version used: version 2, published 2026-06-03
- Source license: CC BY-NC-SA 4.0
- Source catalog size: 131,930 perfumes

The package does not include the raw source catalog or source-site imagery. It contains synthetic
conversations whose evidence cards were constructed from selected structured fields.

ScentAI is an independent research project. It is not affiliated with or endorsed by Fragrantica,
Le Decanteur, perfume houses, or the model/API providers used during synthetic generation.

## License

This derivative dataset is released under **CC BY-NC-SA 4.0** to preserve the attribution,
non-commercial, and share-alike requirements of the source dataset.

You must:

- credit both ScentAI and the Le Decanteur source dataset;
- use the data only for non-commercial purposes;
- distribute adaptations under the same license;
- retain a link to the license and indicate material changes.

License text: https://creativecommons.org/licenses/by-nc-sa/4.0/

The repository source code has separate licensing status. Dataset rights do not automatically grant
rights to trademarks, product imagery, third-party websites, or base-model weights.

## Intended Uses

- instruction tuning for grounded recommendation systems;
- retrieval-augmented generation experiments;
- constraint-following and recommendation evaluation;
- curriculum and synthetic-data research;
- educational, reproducibility, and non-commercial research work.

## Out-of-Scope Uses

- commercial training or resale;
- representing ratings, prices, popularity, or availability as live data;
- medical, allergy, or safety advice;
- guaranteed social outcomes such as compliments or attraction;
- treating synthetic profiles or generated explanations as observed user behavior.

## Limitations

- The corpus inherits omissions, labels, popularity effects, and community biases from its source.
- Ratings and performance values are historical snapshots, not objective laboratory measurements.
- Most examples are English and should not be treated as balanced multilingual coverage.
- Synthetic phrasing can contain recurring stylistic patterns even after duplicate checks.
- The validation split measures generalization within the same synthetic construction process.
- Perfume cards can be long; training requires deliberate sequence-length and truncation handling.
- The two role exports are semantically duplicate representations, not independent datasets.

## Loading

```python
import gzip
import json

with gzip.open("scentai_train_gemma.jsonl.gz", "rt", encoding="utf-8") as handle:
    first_record = json.loads(next(handle))

print(first_record["messages"])
```

With Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "scentai_train_gemma.jsonl.gz",
        "validation": "scentai_validation_gemma.jsonl.gz",
    },
)
```

## Rebuilding The Kaggle Package

From the ScentAI source repository, with the original local `train_set/` available:

```bash
python tools/audit_training_dataset.py \
  --dataset-root /path/to/Perfume-Dataset \
  --output dataset/statistics.json

python tools/build_kaggle_package.py \
  --dataset-root /path/to/Perfume-Dataset \
  --output /path/to/kaggle_release/scentai-32k-grounded-perfume-conversations \
  --owner YOUR_KAGGLE_USERNAME
```

After inspecting `manifest.json`, authenticate with Kaggle and create the dataset privately first.
The current Kaggle CLI requires Python 3.11 or newer and supports browser login:

```bash
kaggle auth login
kaggle datasets create -p /path/to/kaggle_release/scentai-32k-grounded-perfume-conversations
```

Older Kaggle API clients, including `1.7.4.5`, do not provide `kaggle auth login`. For those
versions, create a **Legacy API Key** from the Kaggle API settings page, place the downloaded
`kaggle.json` at the location named by the client's error message, and restrict its permissions:

```bash
mkdir -p ~/.config/kaggle
install -m 600 ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json
kaggle datasets list --mine
```

Never commit `kaggle.json`, an access token, or its contents to this repository.

The public/private setting can then be reviewed on Kaggle before publication. Passing `--public`
to `kaggle datasets create` publishes immediately and should only be used after the package has
been checked.

## Citation

Until a formal archival citation is available, cite the Kaggle dataset URL, repository URL, version,
and access date. Also cite the Le Decanteur source dataset listed above.
