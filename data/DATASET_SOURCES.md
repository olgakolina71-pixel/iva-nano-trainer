# IVA dataset acquisition plan

This is a curated starting set, not a blanket approval of every item on the internet. The importer reads only the explicit allowlist in `import_hf_datasets.py`, streams bounded subsets, filters likely non-Russian records, and records provenance separately per source.

| Dataset | Declared license | Planned role | Status / caveat |
|---|---|---|---|
| [Imperius/ru-classic](https://huggingface.co/datasets/Imperius/ru-classic) | CC0-1.0 | Russian prose pretraining | Dataset card says public-domain source works and CC0 compilation; classic literature only, not modern facts/dialogue; OCR artifacts possible. |
| [nyuuzyou/edutexts](https://huggingface.co/datasets/nyuuzyou/edutexts) | CC0-1.0 | Educational Russian text | Card declares CC0; primarily Russian but includes Ukrainian/English; presentations can contain noisy slide fragments. |
| [SonexaAI/Small-Life-Dataset-ru-eng](https://huggingface.co/datasets/SonexaAI/Small-Life-Dataset-ru-eng) | CC0-1.0 | Small dialogue supplement | Card declares CC0; synthetic short dialogues; repository layout and viewer have shown inconsistencies, so importer may need maintenance. |
| [SonexaAI/ru_eng-dataset](https://huggingface.co/datasets/SonexaAI/ru_eng-dataset) | CC BY 4.0 | Possible later instruction-tuning set | Not included in automatic importer yet; requires attribution and inspection of exact files/splits. |
| [OpenCorpora](https://opencorpora.org/?page=downloads) | CC BY-SA 3.0 | Morphology/lexicon auxiliary resource | Not prose; keep out of language-model pretraining until a specific auxiliary task is implemented. |
| Existing KartaSlov sources | CC BY-NC-SA 4.0 | Lexical/sentiment auxiliary tasks | Non-commercial, share-alike; keep legally separate from CC0 sources and don't use for commercial distribution. |
| [nyuuzyou/ruforum](https://huggingface.co/datasets/nyuuzyou/ruforum) | Card claims CC0 | Not imported | Very large user-generated forum corpus; provenance, privacy and personal-data review needed before any use. |
| Russian National Corpus | Restricted terms | Not allowed for pretraining | Its terms limit use to examples illustrating linguistic phenomena; offline access requires agreement. |

## Run

```bash
pip install -r requirements.txt
python import_hf_datasets.py
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json --out-dir checkpoints
```

The script is not run automatically. Review the generated sample text, counts, duplicate rate, and validation split before a long training run. The one-million-token guard is only a safety floor, not a sufficient corpus size for a useful assistant.

## Rights and quality

Dataset-card license metadata is a claim by the publisher, not an independent legal audit. Keep each source's license and attribution in the manifest. Do not merge the KartaSlov CC BY-NC-SA data into the CC0-only run. Do not ingest the forum corpus or any source whose rights/provenance are unclear. A corpus can be technically downloadable and still be unsuitable for model training.


## Additional dialogue and instruction candidates (quarantine only)

| Dataset | Declared license | Planned role | Status / caveat |
|---|---|---|---|
| [IlyaGusev/ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca) | CC BY 4.0 | Russian instruction SFT | Candidate only; card reports imperfect instruction/output quality, so filter carefully. |
| [zjkarina/matreshka](https://huggingface.co/datasets/zjkarina/matreshka) | CC BY 4.0 | Dialogue and summarization SFT | Candidate only; 8.3K rows, inspect samples and privacy signals. |
| [SonexaAI/ru_eng-dataset](https://huggingface.co/datasets/SonexaAI/ru_eng-dataset) | CC BY 4.0 | Short bilingual prompt-response SFT | Candidate only; Dataset Viewer did not expose rows during review; inspect raw JSONL. |
| [DatasetsEval/RusLang-Edu-100](https://huggingface.co/datasets/DatasetsEval/RusLang-Edu-100) | CC BY 4.0 | Russian grammar evaluation / small SFT supplement | Candidate only; 100 rows, not a pretraining corpus. |
| [lksy/ru_instruct_gpt4](https://huggingface.co/datasets/lksy/ru_instruct_gpt4) | CC BY 4.0 | Russian instruction SFT | Hold; viewer disabled for custom code, inspect files and provenance first. |
| [samurainisa/Russian-PD](https://huggingface.co/datasets/samurainisa/Russian-PD) | Not declared in Hub metadata | Russian prose pretraining | Hold; 8.6 GB and item-level rights/schema review needed. |
| [MarkProMaster229/Russian_classical_literature](https://huggingface.co/datasets/MarkProMaster229/Russian_classical_literature) | CC0 1.0 (declared) | Classical literature | Hold; likely duplicates existing classics, verify editions and deduplicate. |

Candidate means eligible for a quarantined snapshot download only. It does not authorize training. Hold sources are blocked by the download workflow.
