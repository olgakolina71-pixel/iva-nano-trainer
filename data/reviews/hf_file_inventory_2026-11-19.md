# Hugging Face file inventory for IVA Nano — 2026-11-19

**Scope:** read-only Hugging Face Hub file listings for all 29 registry sources. No corpus files were downloaded or changed. This inventory supplements the 2026-11-17 content/readiness audit; it is not a license approval or a full content audit.

| # | Registry ID | Hub source | Current file layout / key sizes | Cleanup gate |
|---:|---|---|---|---|
| 1 | `imperius_ru_classic` | [Imperius/ru-classic](https://huggingface.co/datasets/Imperius/ru-classic) | corpus_clean.norm.txt:910421439; corpus_clean.txt:908778710; spm.model:334679; spm.vocab:321516 | Large literary text; edition rights and OCR cleanup remain unresolved. |
| 2 | `sonexa_small_life_ru_eng` | [SonexaAI/Small-Life-Dataset-ru-eng](https://huggingface.co/datasets/SonexaAI/Small-Life-Dataset-ru-eng) | train.jsonl:8911884 | CC0 claim; prior sample showed mixed RU/EN, duplicates and a malformed technical answer; creator metadata unresolved. |
| 3 | `sonexa_medium_life_ru_eng` | [SonexaAI/Medium-Life-Dataset-ru-eng](https://huggingface.co/datasets/SonexaAI/Medium-Life-Dataset-ru-eng) | train.jsonl:193142035; test.jsonl:13873295 | Keep test split separate; previous Russian sample dominated by generic boilerplate. |
| 4 | `kukunechka_russian_everyday_dialogues` | [kukunechka/russian-everyday-dialogues](https://huggingface.co/datasets/kukunechka/russian-everyday-dialogues) | russian_everyday_dialogues.json:8495; russian_everyday_dialogues.jsonl:6538 | 20 short dialogues; neutralize unsupported local prices/routes/services; CC BY attribution. |
| 5 | `atbuba_ruwiki` | [atBuba/ruwiki-dataset](https://huggingface.co/datasets/atBuba/ruwiki-dataset) | ruwiki_full.txt:8425293658; train.bin:2390457572; val.bin:12012348 | Large text plus opaque token binaries; CC BY-SA obligations, source-date and split provenance. |
| 6 | `nyuuzyou_edutexts` | [nyuuzyou/edutexts](https://huggingface.co/datasets/nyuuzyou/edutexts) | documents.parquet:136154733; presentations.parquet:168062048 | Hold for personal-data and upstream document-rights review. |
| 7 | `nyuuzyou_ruforum` | [nyuuzyou/ruforum](https://huggingface.co/datasets/nyuuzyou/ruforum) | 11 JSONL.ZST shards; sizes 304586778–388398097 each | Hold; 58M+ user posts, privacy, safety, copyright and provenance screening required. |
| 8 | `inkoziev_conversations` | [inkoziev/Conversations](https://huggingface.co/datasets/inkoziev/Conversations) | Conversations.py:1934; conversations.jsonl.gz:1574505230 | Do not execute custom loader; inspect code statically and parse raw gzip in a sandbox only after rights/privacy review. |
| 9 | `ilyagusev_ru_turbo_alpaca` | [IlyaGusev/ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca) | ru_turbo_alpaca.jsonl.zst:14565995; ru_turbo_alpaca.py:2566 | Hold for explicit competing-model restriction and answer quality; no training until legal terms resolved. |
| 10 | `zjkarina_matreshka` | [zjkarina/matreshka](https://huggingface.co/datasets/zjkarina/matreshka) | data/ directory; README.md | Parquet train/validation; role/dialog alignment, empty turns, persona, summaries and facts need row-level checks. |
| 11 | `sonexa_ru_eng_prompt_response` | [SonexaAI/ru_eng-dataset](https://huggingface.co/datasets/SonexaAI/ru_eng-dataset) | train.jsonl:9346445; validation.jsonl:158437 | Hold; RU/EN mix, vague prompts, generic answers and duplicates. |
| 12 | `datasets_eval_ruslang_edu_100` | [DatasetsEval/RusLang-Edu-100](https://huggingface.co/datasets/DatasetsEval/RusLang-Edu-100) | Dataset.parquet:60531 | Only 100 rows; inspect all; preserve row-level 90/10 train/validation split; CC BY attribution. |
| 13 | `lksy_ru_instruct_gpt4` | [lksy/ru_instruct_gpt4](https://huggingface.co/datasets/lksy/ru_instruct_gpt4) | ru_instruct_gpt4.jsonl:23276814; ru_instruct_gpt4.py:1208 | Do not execute custom loader; source rights, privacy and quality not established. |
| 14 | `samurainisa_russian_pd` | [samurainisa/Russian-PD](https://huggingface.co/datasets/samurainisa/Russian-PD) | 9 Parquet shards; sizes 152341835–1560838613 each | No Hub license tag; large Parquet files; item-level public-domain and edition provenance required. |
| 15 | `markpromaster_russian_classical_literature` | [MarkProMaster229/Russian_classical_literature](https://huggingface.co/datasets/MarkProMaster229/Russian_classical_literature) | MydatasetT2.json:6721208 | Potential overlap with other classical corpora; edition-level rights and cross-corpus deduplication. |
| 16 | `openai_gsm8k_math` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | main/ and socratic/ configs | English math only; tests remain evaluation-only; no automatic translation. |
| 17 | `tigerlab_mathinstruct` | [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | MathInstruct.json:212488891 | Mixed upstream sources/licenses; row-level license filter; math correctness and duplicates. |
| 18 | `microsoft_orca_math_200k` | [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | data/ directory | English generated math; independently verify calculations; do not silently translate. |
| 19 | `convai_physics_reasoning` | [convaiinnovations/physics-reasoning-dataset](https://huggingface.co/datasets/convaiinnovations/physics-reasoning-dataset) | physics_dataset_streaming.jsonl:9460076; train.parquet:3536025 | English; group prompt/paraphrase/counterfactual variants before split; verify units and dimensions. |
| 20 | `allenai_sciq` | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | data/ directory | CC BY-NC 3.0; English; not for general training under current project policy. |
| 21 | `maximus_physicsqa` | [maximus-21/PhysicsQA](https://huggingface.co/datasets/maximus-21/PhysicsQA) | PhysicsQA.json:304185 | Online-source provenance unclear; verify upstream rights and answers. |
| 22 | `flytech_python_codes_25k` | [flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k) | python-codes-25k.json:26356325; python-codes-25k.jsonl:25445912 | Hub license tag absent; reconcile 25K naming and file rows; syntax/security tests and source rights. |
| 23 | `d0rj_math_dataset_ru` | [d0rj/mathematics_dataset](https://huggingface.co/datasets/d0rj/mathematics_dataset) | data/ and rl/ directories | 12.7M Russian rows; train-easy/medium/hard only; interpolate/extrapolate evaluation-only; verify answers. |
| 24 | `s85io_math_logic_ru` | [s85io/math-logic-ru](https://huggingface.co/datasets/s85io/math-logic-ru) | train.jsonl:6746557; validation.jsonl:733113 | Known invalid inference; symbolic proof validation and human review required. |
| 25 | `fmluder_tiny_russian_dialogs` | [Fmluder/Tiny-Russian-Dialogs](https://huggingface.co/datasets/Fmluder/Tiny-Russian-Dialogs) | Tiny-Russian-Dialogs.csv:23198725 | Prior sample had boilerplate, repeated greetings and malformed Russian; full dedup/relevance review. |
| 26 | `ttech_tmath_ru_eval` | [t-tech/T-math](https://huggingface.co/datasets/t-tech/T-math) | data/ directory | Evaluation-only forever; card count 331 vs current Parquet listing 325; never train on it. |
| 27 | `ru_dataset_small_hold` | [ru-dataset/ru-dataset-small](https://huggingface.co/datasets/ru-dataset/ru-dataset-small) | dataset-1/ through dataset-9/ directories | Limited schema; reasoning-format and source-quality review; don't execute custom scripts. |
| 28 | `attn_signs_russian_code_hold` | [attn-signs/russian-code](https://huggingface.co/datasets/attn-signs/russian-code) | data/ directory | No declared license; blocked until rights are documented. |
| 29 | `sup2ch_ru_en_code_curriculum_hold` | [sup2ch/ru-en-code-curriculum](https://huggingface.co/datasets/sup2ch/ru-en-code-curriculum) | data/, metadata/, samples/, ATTRIBUTION.md, LICENSE.md, SHA256SUMS.json | Per-row license/source/revision/redistributability filters; exclude reserve/validation from training. |

## Current decision
- Registry: 29 sources; 5 candidates for further quarantine inspection; 24 holds.
- Training-approved sources: 0.
- `training_enabled`: false.
- The file listing confirms the current storage formats and reveals that several sources are multi-gigabyte, compressed, Parquet-backed, or use custom Python loaders.
- Custom loader files (`Conversations.py`, `ru_turbo_alpaca.py`, `ru_instruct_gpt4.py`, `change_the_system_prompt.py`) must not be executed as part of an audit. Inspect their source statically and use a safe parser for data files.
- Large forum, educational and classical-literature corpora require streaming and privacy/rights review; they must not be bulk-loaded into memory or training automatically.
- This inventory does not claim any dataset has been cleaned row-by-row. The all-29 workflow is prepared to produce bounded aggregate sample diagnostics only; it still needs an authenticated GitHub Actions dispatch.

## Safety invariants
1. Never include the t-tech/T-math benchmark in training.
2. Keep validation/test partitions separate, including the 90/10 row-level split in RusLang-Edu-100.
3. For d0rj mathematics, train only on train-easy/train-medium/train-hard; retain interpolate/extrapolate for evaluation.
4. Apply row-level upstream license filtering to TIGER-Lab/MathInstruct and sup2ch/ru-en-code-curriculum.
5. Do not treat publisher license tags as proof of upstream rights.
6. Never publish raw samples containing personal data; keep only aggregate audit diagnostics.
