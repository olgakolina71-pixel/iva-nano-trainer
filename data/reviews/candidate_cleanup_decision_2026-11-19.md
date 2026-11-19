# IVA Nano — candidate cleanup and promotion decision (2026-11-19)

## Result

A cleaned, 20-pair SFT derivative was prepared for `kukunechka/russian-everyday-dialogues`:

- Cleaned file: `data/sft_candidates/kukunechka_russian_everyday_dialogues_clean_v1.txt`
- Source: https://huggingface.co/datasets/kukunechka/russian-everyday-dialogues
- Source card declares CC BY 4.0 and names Ekaterina Myskina in its citation metadata.
- Attribution must be retained in any distribution or training records.

This is a **quarantine-stage cleaned derivative**, not an approved training source.

## Cleaning performed

All 20 source pairs were retained and normalized to IVA's SFT turn format. The following unsupported claims were neutralized rather than treated as factual knowledge:

- Store-specific prices, payment methods, discount-card setup and store layout.
- Specific public-transport routes, stops and waiting times.
- Restaurant wait-time promises and unverified menu availability.
- Freshness/date-of-delivery claims about food.
- Gluten-free suitability: the revised response explicitly asks staff to verify ingredients and cross-contact.
- A blanket 14-day return claim: the revised response directs the user to the applicable store/product terms.

Punctuation was corrected in the transport dialogue. No new local prices, routes, schedules, legal rules or service promises were invented.

## Why it is not promoted yet

1. The Hub card declares CC BY 4.0, but the uploader's authority over all underlying material has not been independently established.
2. The source contains only 20 dialogue pairs. It is a tiny style supplement, not a meaningful standalone SFT corpus.
3. The project SFT pipeline requires at least 200,000 supervised assistant tokens; this file is far below that threshold.
4. The maintainer must explicitly approve the source and attribution before it can be listed as `use: training` with `rights_status: reviewed_approved`.

## Remaining candidate decisions

| Candidate | Decision |
|---|---|
| `kukunechka/russian-everyday-dialogues` | Cleaned derivative prepared; pending rights/maintainer approval; tiny SFT supplement only |
| `DatasetsEval/RusLang-Edu-100` | Keep 90 row-level train records separate from 10 validation records; not approved until answers/explanations and row-level split are independently checked |
| `d0rj/mathematics_dataset` | Promising Russian math source; not approved until stratified checks cover all modules and answers; only train-easy/medium/hard may be considered |
| `atBuba/ruwiki-dataset` | Not approved; 8.4 GB text snapshot, file-size/provenance discrepancy, CC BY-SA obligations and privacy checks remain |
| `t-tech/T-math` | Evaluation-only; never add to pretraining or SFT |

The other 24 registry entries remain on hold. No candidate is marked approved in this report.

## Training gate

- `data/manifest.json` was not created or modified.
- `data/hf_manifest.json` was not modified.
- No candidate was added with `use: training`.
- `training_enabled` remains `false`.
- No training job was started.
- No raw multi-gigabyte dataset was copied into GitHub.
