# IVA Nano Trainer

A separate Python/PyTorch training project for the IVA Nano language model.

## Current status

This repository contains a training scaffold and a **Cloudflare dataset import pipeline**. It is not a trained 40M-parameter model and has no hosted GPU runtime. The Cloudflare Worker is the API/control plane; PyTorch training runs separately.

## Import approved datasets

The importer reads the Worker catalog, checks the allowlist, usage status, exact license and attribution, downloads the CSV through the Worker proxy, validates UTF-8/CSV structure, calculates SHA-256, and writes a local manifest.

```bash
pip install -r requirements.txt
python import_datasets.py
```

Optional selection:

```bash
python import_datasets.py --only assoc_safe kartaslovsent
```

Files are written to `data/imported/`. The importer does not bypass the Cloudflare license gate and rejects missing or mismatched license metadata.

**Important data distinction:** these four sources are not all general-purpose text corpora. Association and sentiment datasets are task-oriented; the Open Semantics hierarchy and Orfo letter matrix are auxiliary structured data. Imported CSVs are therefore marked `auxiliary_only_pending_schema_review` and are **not automatically fed into language-model pretraining**. Each schema needs a reviewed conversion/learning objective first.

All four currently carry CC BY-NC-SA 4.0 metadata. This is a non-commercial, share-alike license with attribution requirements. Keep the project non-commercial unless the rights situation is separately reviewed. The importer records attribution and checks the license label, but that is not legal advice or proof of upstream ownership.

## Prepare auxiliary Russian examples

After importing the approved CSVs, convert only the association and sentiment tables into attributed Russian training lines:

```bash
python prepare_auxiliary_text.py --assoc-csv data/imported/assoc_safe.csv --sentiment-csv data/imported/kartaslovsent.csv
```

This writes `data/raw/kartaslov_auxiliary.txt` and `data/manifest.json`, which match the trainer defaults. The converter expects semicolon-delimited `word/assoc/is_safe` and `term/tag` fields. It deliberately does **not** turn the semantic hierarchy or letter-frequency matrix into fake prose. The generated lines teach simple lexical associations and sentiment labels; they are a small auxiliary dataset, not enough to produce a capable conversational model. The source license remains CC BY-NC-SA 4.0, so training and derivatives must respect non-commercial and share-alike terms.

## Corpus quality gate

Full IVA Nano training now refuses to start unless the tokenized corpus contains at least 100,000,000 tokens (`training.minimum_training_tokens`). This is only a minimum engineering guardrail, not enough by itself to make a capable assistant. The initial research target is 800,000,000 tokens, with quality measured on a held-out validation set.

A candidate Russian-language source is [Imperius/ru-classic](https://huggingface.co/datasets/Imperius/ru-classic). Its dataset card declares CC0 for the compilation and describes 866 MB of cleaned Russian classical literature from 61 authors. It is useful for learning Russian prose style, but the card explicitly warns that it is not intended for modern factual QA and may contain rare OCR artifacts. Review the dataset card and underlying rights before use; it is not a modern conversation corpus. Do not mix it with the existing CC BY-NC-SA KartaSlov data without preserving each source's separate license and attribution metadata.

## Import additional Russian datasets

A separate manual workflow is available at **Actions → IVA dataset acquisition → Run workflow**. It streams bounded subsets of three dataset-card-CC0 sources into separate UTF-8 text files and produces `data/hf_manifest.json` with per-source provenance. The workflow uploads the snapshots and manifest as a 7-day artifact. It does not train IVA Nano.

To run locally:

```bash
pip install -r requirements.txt
python import_hf_datasets.py
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json --out-dir checkpoints
```

Read [data/DATASET_SOURCES.md](data/DATASET_SOURCES.md) before use. The importer keeps the sources separate and does not include the KartaSlov non-commercial dataset, forum posts, or sources with unclear rights.

## Train

Put reviewed, permitted UTF-8 text files under `data/raw/` and create `data/manifest.json` with license and attribution for every source. Then:

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw --out-dir checkpoints
```

The CSV import directory is separate from `data/raw/` to prevent accidental training on unreviewed structured files.

## Run the first real diagnostic training

A manual GitHub Actions workflow is available at **Actions → IVA Micro data-and-training smoke run → Run workflow**. It imports only the two reviewed lexical datasets (associations and sentiment), converts them into attributed Russian examples, trains a deliberately tiny CPU model for 20 steps, generates one diagnostic continuation, and uploads the checkpoint/tokenizer as a short-lived artifact (7 days).

This is a pipeline test, **not IVA Nano 40M training** and not proof of fluent Russian. Its purpose is to verify end-to-end data import, tokenizer creation, optimization, checkpoint saving, and generation without requiring R2 or a GPU. The workflow is manual so it does not consume runner time on every code change.

## Generate text

After a checkpoint and tokenizer have been trained, run:

```bash
python generate.py --prompt "Привет! Расскажи о себе:" --max-new-tokens 80 --temperature 0.8 --top-k 40
```

Sampling controls reduce uncontrolled randomness, but they cannot create language understanding. Coherent output depends on a sufficiently large, clean, licensed text corpus and successful training. The four imported CSVs are not yet converted into general pretraining text, so they do not by themselves make IVA a fluent conversational model.

## Model

- Decoder-only Transformer, configurable near 40M parameters.
- SentencePiece BPE tokenizer.
- Local text training and local checkpoints.
- GitHub Actions runs code checks and a small CPU model smoke test only; it is not a free GPU training service.

## Cloudflare integration

Worker catalog: `/api/datasets`; CSV proxy: `/api/datasets/{id}/raw`. The importer uses these existing routes. Do not put API tokens in Git. **No R2 dependency:** IVA is designed to work without Cloudflare R2 for the next several years. Dataset snapshots stay in the trainer's `data/imported/` directory; checkpoints are saved to a persistent filesystem volume. Set `IVA_ARTIFACT_DIR` to that volume's path. The local artifact store writes checkpoints atomically and records SHA-256 checksums in `artifact_manifest.json`. Do not use GitHub commits or D1 as a substitute for large binary storage: D1 is for metadata, and GitHub Actions runner disks are temporary. If a training machine is ephemeral, copy artifacts to persistent storage before it shuts down. Workers remain the API/control plane, not PyTorch pretraining machines. No R2 bucket, billing activation, or payment card is required.


## IVA Nano 40M: staged training

The Nano configuration now targets **40,015,008 trainable parameters** (528-wide, 8-layer decoder-only Transformer, 8 attention heads, 24k SentencePiece vocabulary, 1,024-token context). The training config is intentionally separate from the 173k-parameter Micro diagnostic.

The full-pretraining guard is set to 100 million tokens as a minimum engineering gate; the initial research target is 800 million tokens. These are not guarantees of quality. A 40M model trained from random initialization needs a diverse, clean corpus and substantially more compute than the 20-step CPU smoke test. Use a CUDA-capable machine for a real run. GitHub-hosted CPU Actions are for validation and smoke tests, not the full pretraining job.

### Rights review is required before training

The Hugging Face acquisition workflow now writes each downloaded dataset with `use: pending_rights_review` and `rights_status: dataset_card_claim_unverified`. This is deliberate: a dataset card's CC0 claim is not independent proof of the rights to every underlying item. The standard loader ignores these pending entries. Review source provenance and terms first; only then may a maintainer explicitly change the relevant manifest entry to `use: training`. Keep each source's own license and attribution. Do not mix the KartaSlov CC BY-NC-SA material into a commercial model or remove its share-alike obligations.

The dialogue source is converted into explicit `### Пользователь:` / `### IVA:` turns, suitable for a later instruction-tuning stage. It remains quarantined until rights review.

### Pretraining

After adding enough reviewed, diverse Russian text and approving its manifest entries:

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json --out-dir checkpoints
```

The default Nano run refuses corpora below 100M tokens. The configured schedule is 100,000 optimizer steps. Do not launch this on a short-lived CPU runner and expect a useful model.

### Instruction fine-tuning

After a Nano base checkpoint exists, first review the dialogue source's provenance and terms. In `data/hf_manifest.json`, set that one source entry to `use: training` and `rights_status: reviewed_approved` only after the review is complete. Then prepare the separate SFT corpus:

```bash
python prepare_sft.py --manifest data/hf_manifest.json --dataset-id small_life_dialogues_cc0
```

After a Nano base checkpoint exists and the reviewed dialogue/instruction corpus is prepared, run:

```bash
python train.py --config configs/iva_nano_sft.yaml --data-dir data/sft --manifest data/sft/manifest.json --out-dir checkpoints/sft --tokenizer-path checkpoints/iva_tokenizer.model --resume checkpoints/artifacts/iva_nano_40m_checkpoint.pt --reset-optimizer --additional-steps
```

This loads the base weights, keeps the original tokenizer, and starts a fresh lower-learning-rate optimizer. The SFT manifest must list only reviewed, attributed training text files inside `data/sft/`.

### Chat locally

After training, place the checkpoint and matching tokenizer at the paths below, then run:

```bash
python chat.py
```

Or use explicit paths:

```bash
python chat.py --checkpoint checkpoints/artifacts/iva_nano_40m_checkpoint.pt --tokenizer checkpoints/iva_tokenizer.model
```

The CLI supports multi-turn history, `/reset`, and `/exit`. It is an inference interface, not a substitute for instruction fine-tuning. A base-pretrained model may continue text without reliably following instructions.

### Resume a pretraining run

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json --out-dir checkpoints --tokenizer-path checkpoints/iva_tokenizer.model --resume checkpoints/artifacts/iva_nano_40m_checkpoint.pt --additional-steps
```

Checkpoints include model ID, architecture, optimizer state, step count, parameter count, and tokenizer path. Set `IVA_ARTIFACT_DIR` to a persistent local volume; GitHub Actions artifacts expire after 7 days. No Cloudflare R2 or paid storage is required.

## Honest project status

The repository now contains the staged model/training/inference pipeline, but **there is still no trained IVA Nano checkpoint**. The previous successful run trained only IVA Micro (173,312 parameters) for 20 CPU steps on lexical examples. It did not pretrain Nano or verify conversational quality. A real Nano model will exist only after a long GPU training run on a rights-reviewed corpus, followed by instruction tuning and evaluation.
