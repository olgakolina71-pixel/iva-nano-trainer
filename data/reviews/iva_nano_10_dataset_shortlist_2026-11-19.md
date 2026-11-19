# IVA Nano — 10 dataset shortlist (2026-11-19)

This is a **candidate list, not training approval**. It is a practical shortlist to reach the goal of at least 10 sources, subject to pipeline compatibility, sample review, license/provenance review, deduplication, PII screening, and source-level split preservation. Do not set `training_enabled=true` based on this file.

## Candidate sources

| # | Dataset | Proposed role | Declared license | Status / caveat |
|---|---|---|---|---|
| 1 | [Imperius/ru-classic](https://huggingface.co/datasets/Imperius/ru-classic) | Russian text pretraining | CC0-1.0 (card claim) | Owner accepted card declaration for this run; provenance still not independently verified. |
| 2 | [SiberiaSoft/SiberianDataset](https://huggingface.co/datasets/SiberiaSoft/SiberianDataset) | Russian chat, QA, instructions | MIT (card claim) | HOLD: mixture contains upstream sources; audit provenance, privacy, quality, overlap. |
| 3 | [ZeroAgency/ru-big-russian-dataset](https://huggingface.co/datasets/ZeroAgency/ru-big-russian-dataset) | Russian instruction/chat/math/code | MIT (card claim) | HOLD: large mixture; inspect sources, train split only, exclude test, audit PII and overlap. |
| 4 | [ru-dataset/ru-dataset-small](https://huggingface.co/datasets/ru-dataset/ru-dataset-small) | Russian coding/math/instruction SFT | Apache-2.0 (card claim) | HOLD: small; sample answer review and schema conversion. |
| 5 | [sup2ch/ru-en-code-curriculum](https://huggingface.co/datasets/sup2ch/ru-en-code-curriculum) | Russian-English code CPT/SFT | Multi-license, row-level | HOLD: only rows marked redistributable; respect train/reserve/validation and per-row license; use train only. |
| 6 | [SonexaAI/Medium-Life-Dataset-ru-eng](https://huggingface.co/datasets/SonexaAI/Medium-Life-Dataset-ru-eng) | RU/EN dialogue + coding | CC-BY-4.0 (card claim) | HOLD: publisher describes synthetic data; verify files/schema, quality, attribution. |
| 7 | [SonexaAI/ru_eng-humanity-dataset](https://huggingface.co/datasets/SonexaAI/ru_eng-humanity-dataset) | RU/EN synthetic instruction/dialogue | CC0-1.0 (card claim) | HOLD: inspect examples, duplicates, generator artifacts, split integrity. |
| 8 | [SonexaAI/ru_eng-dataset](https://huggingface.co/datasets/SonexaAI/ru_eng-dataset) | Short RU/EN prompt-response dialogue | CC-BY-4.0 (card claim) | HOLD: viewer/export errors; inspect raw files and split before considering. |
| 9 | [LLiserginov/russian-instructions-10k](https://huggingface.co/datasets/LLiserginov/russian-instructions-10k) | Russian instructions, coding/math | CC-BY-NC-4.0 / upstream terms (card claim) | HOLD/likely exclude from commercial-compatible training; translated Alpaca lineage and terms require review. |
| 10 | [its5Q/resh-edu](https://huggingface.co/datasets/its5Q/resh-edu) | Russian school education text/QA | CC0-1.0 (card claim) | HOLD: scraped website source and custom loader; do not execute loader. Inspect data-only files and provenance first. |

## Additional small language-specific candidate
[RusLang-Edu-100](https://huggingface.co/datasets/DatasetsEval/RusLang-Edu-100) is a conditional candidate, not one of the ten. Two answer issues were found (rl-000088, rl-000089); the remaining 88 train rows have not all been manually checked. Preserve its 90/10 row-level split. Not approved.

## Proposed mix (after audits)
- Pretraining: primarily #1 plus only explicitly eligible clean text sources after provenance and privacy review.
- SFT: selected verified train records from #2–#8, converted to a common message schema.
- #9 is not selected for now because of its non-commercial restriction/lineage.
- #10 is not selected until source rights are clarified.
- Keep all test/validation/reserve splits out of training; never use benchmark/test records for training.
- Deduplicate across all selected corpora, not just within each source; retain source ID, revision, license, split, and hash.

## Required gate before activation
1. Pin exact dataset revision for each source and record file hashes.
2. Use only safe data-only loading paths; do not execute custom dataset scripts.
3. Verify the intended license and upstream source terms.
4. Inspect schema and samples; audit language, answer quality, duplication, and PII.
5. Preserve original splits; only eligible training splits may be used.
6. Build a merged, source-attributed staging manifest; report eligible token count.
7. Only after maintainer approval, add selected sources to the training manifest.

**Current state:** this document does not activate any source. Keep `data/hf_manifest.json` and the training workflow's approval gate unchanged until the audits are complete.
