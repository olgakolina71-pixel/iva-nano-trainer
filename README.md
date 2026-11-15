# IVA Nano Trainer

A separate Python/PyTorch training project for the IVA Nano language model.

## Current status

This repository contains the **initial training scaffold**, not a trained 40M-parameter model. It does not contain model weights or a GPU runtime. The Cloudflare Worker remains the API/control plane; model pretraining must run on a separate machine or compute service.

## Design

- Decoder-only Transformer (GPT-style), configurable near 40M parameters.
- SentencePiece BPE tokenizer.
- Text-only training from local UTF-8 `.txt` files.
- Dataset license/attribution manifest required before preparing data.
- Checkpoints and metrics written locally.
- GitHub Actions performs code and CPU smoke checks only; it is not a free GPU training service.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare data

Put permitted UTF-8 text files under `data/raw/`. Review the dataset terms yourself and record the source, license, attribution, and permitted use in `data/manifest.json`. Do not add private, personal, or unlicensed material.

Example manifest:

```json
{
  "datasets": [
    {
      "id": "example-corpus",
      "path": "data/raw/example.txt",
      "license": "CC BY 4.0",
      "attribution": "Dataset author and source URL",
      "use": "training"
    }
  ]
}
```

The trainer refuses to run if the manifest is missing or a dataset entry has no license or attribution. This is a basic safeguard, not legal advice.

## Train

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw --out-dir checkpoints
```

The first run trains a SentencePiece tokenizer from the local text files. Keep datasets and checkpoints out of Git.

## Cloudflare integration

The Cloudflare Worker should coordinate status and API requests; it should not run PyTorch pretraining inside a Worker. A future authenticated callback can publish run status and checkpoint metadata. Never commit API tokens or secrets.

## Important

The current scaffold is not connected to the Cloudflare dataset proxy yet. Dataset import, checksum verification, remote checkpoint storage, resumable jobs, and a secure Worker callback are planned follow-up tasks. No paid service or payment card is required to inspect or run the CPU smoke test.
