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

## Train

Put reviewed, permitted UTF-8 text files under `data/raw/` and create `data/manifest.json` with license and attribution for every source. Then:

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw --out-dir checkpoints
```

The CSV import directory is separate from `data/raw/` to prevent accidental training on unreviewed structured files.

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
