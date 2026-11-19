# IVA Nano — dataset shortlist and training gate (2026-11-19)

## Decision

This is a research shortlist, NOT a training approval. No dataset is certified as "100% quality"; no external dataset is approved by this document. Keep `training_enabled: false` and do not add any source below to the training manifest until the existing project audit gate is satisfied.

No corpus binaries are stored in GitHub. This file stores source links and handling decisions only.

## Best large-scale pretraining candidate

### HuggingFaceFW/fineweb-2
- Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- Declared license: ODC-By 1.0 (verify applicable obligations before redistribution/use).
- Scope: multilingual web pretraining corpus with 1,868 language-script pairs.
- Publisher describes language identification, per-language deduplication, filtering, and PII anonymization.
- The dataset card describes a reproducible pipeline and reports roughly 20 TB overall, 5 billion documents, and more than 3 trillion words.
- Russian subset: exists as a language-script subset, but the exact current size and usable shard list must be queried before downloading. Do not assume the full corpus or a previously quoted size is correct.
- Role: pretraining text only; it is not instruction/dialogue SFT.
- Remaining checks: inspect Russian subset schema and file list; pin revision and shard hashes; preserve the official test split outside training; review source terms and opt-out/removal process; run project-level sample, PII, encoding, language and duplication checks.
- Decision: HOLD — strongest documented multilingual pretraining candidate, but not "no-check" data.

## Other useful sources (not substitutes for Russian SFT)

### openai/gsm8k
- Source: https://huggingface.co/datasets/openai/gsm8k
- Hub metadata: MIT; English; approximately 7.5K train and 1.3K test rows per config.
- Role: small math word-problem SFT/evaluation source.
- Rule: train split only; keep every test split evaluation-only. The source is English; do not silently machine-translate it into Russian.
- Remaining checks: verify arithmetic/solution consistency and pin revision.
- Decision: HOLD pending answer verification; existing registry entry remains authoritative.

### microsoft/orca-math-word-problems-200k
- Source: https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
- Hub metadata: MIT; English; 200,000 train rows; question/answer schema.
- Publisher says answers were generated with Azure GPT-4 Turbo.
- Role: math word-problem SFT, not Russian-language pretraining.
- Remaining checks: independently verify math and reasoning, inspect duplicates and source revision.
- Decision: HOLD pending answer verification; existing registry entry remains authoritative.

### allenai/dolma
- Source: https://huggingface.co/datasets/allenai/dolma
- Hub metadata: ODC-BY; English; size category above one trillion tokens.
- AllenAI describes Dolma as an open English pretraining corpus used for OLMo.
- Role: English pretraining only; not a Russian corpus.
- Remaining checks: confirm the exact version, license obligations, and source revision before use.
- Decision: NOT selected for Russian-first IVA training.

## Why no dataset is approved automatically

A Hub license tag and a polished dataset card are useful evidence, not a guarantee that every upstream item is correctly licensed or that every answer is correct. Dataset cards are intended to document source, license, language, size, intended use and limitations. FineWeb-2 itself documents substantial filtering and anonymization, but that does not make it error-free or remove the need to follow its use conditions.

Before any source is moved into `data/manifest.json` with `use: training`, complete:
1. Pin exact revision/commit and record shard checksums.
2. Confirm license and upstream provenance for the intended use.
3. Inspect schema, language distribution and sample quality.
4. Screen personal data and sensitive content.
5. Deduplicate and check train/eval leakage.
6. For math/physics/code, validate answers or run suitable automated checks plus human spot-checks.
7. Preserve attribution and required notices.
8. Record a maintainer approval for the exact source revision and allowed split.

## Project safety state

- `data/dataset_candidates.json`: 29 registered sources at time of writing.
- `data/hf_manifest.json`: external candidates remain pending/held.
- `training_enabled`: must remain `false`.
- Do not train on benchmark/test splits, including GSM8K test or T-math.
- Do not upload multi-gigabyte source corpora or downloaded shards to GitHub. Keep GitHub for code, manifests, audit reports and checksums; use the dataset host or a dedicated object store for data files.

## Documentation references

- Hugging Face dataset cards: https://huggingface.co/docs/hub/datasets-cards
- Hugging Face dataset viewer / dataset information: https://huggingface.co/docs/dataset-viewer/en/info
- FineWeb-2 dataset card and processing details: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- Dolma documentation: https://docs.allenai.org/training_data/dolma/index.html
