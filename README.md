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

A separate manual workflow is available at **Actions → IVA candidate dataset acquisition (quarantine) → Run workflow**. It streams bounded subsets of three candidate sources into separate UTF-8 text files and produces `data/hf_manifest.json` with per-source provenance. All entries remain `pending_rights_review`; the dataset-card license declarations are not independent proof of rights. The workflow uploads the snapshots and manifest as a 7-day artifact. It does not train IVA Nano.

To run locally:

```bash
pip install -r requirements.txt
python import_hf_datasets.py
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json --out-dir checkpoints
```

**Do not run the training command while the manifest entries are pending.** First review source provenance, underlying content rights, and quality; only then explicitly mark an approved entry `use: training` and `rights_status: reviewed_approved`. The importer intentionally does not make this decision.

Read [data/DATASET_SOURCES.md](data/DATASET_SOURCES.md) before use. The importer keeps the sources separate and does not include the KartaSlov non-commercial dataset, forum posts, or sources with unclear rights.

## Normalize candidate datasets (quarantine-safe)
The first acquired snapshot was normalized and measured; see [the corpus and privacy-screening report](reports/candidate-corpus-2026-11-15.json). The report records 12,558 records flagged by heuristic personal-data checks and a major dialogue-source discrepancy: the dataset card says 50,000 pairs, but this snapshot yielded only 105. This is not legal or privacy clearance.


After downloading the acquisition artifact and extracting its `raw/hf/` files into the repository, normalize the JSON-encoded educational rows, preserve multi-line dialogue turns, remove exact duplicate records, flag obvious student-name/class identifiers, email addresses and phone numbers, and write a separate manifest:

```bash
python prepare_candidate_corpus.py --input-dir data/raw/hf --input-manifest data/hf_manifest.json --output-dir data/normalized_candidates --output-manifest data/normalized_candidates/manifest.json
```

This step does **not** approve any dataset. It preserves each source's license, attribution, `use`, and `rights_status`; all sources currently remain quarantined. The output is for inspection and corpus-quality analysis only until rights are independently reviewed. It does not train a model. Automated personal-data checks are only a first filter; they do not replace a human privacy review.

## Train

Put reviewed, permitted UTF-8 text files under `data/raw/` and create `data/manifest.json` with license and attribution for every source. Then:

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw --out-dir checkpoints
```

The CSV import directory is separate from `data/raw/` to prevent accidental training on unreviewed structured files.

## Run the first real diagnostic training

A manual GitHub Actions workflow is available at **Actions → IVA Micro data-and-training smoke run → Run workflow**. It imports only the two reviewed lexical datasets (associations and sentiment), converts them into attributed Russian examples, trains a deliberately tiny CPU model for 200 steps, generates one diagnostic continuation, and uploads the checkpoint/tokenizer as a short-lived artifact (7 days).

This is a pipeline test, **not IVA Nano 40M training** and not proof of fluent Russian. Its purpose is to verify end-to-end data import, tokenizer creation, optimization, checkpoint saving, and generation without requiring R2 or a GPU. The workflow is manual so it does not consume runner time on every code change.

## Evaluate a trained checkpoint

After a checkpoint and its matching tokenizer exist, evaluate it on a deterministic held-out 5% split of the approved corpus and save sample generations:

```bash
python evaluate.py --checkpoint checkpoints/artifacts/iva_nano_40m_checkpoint.pt --tokenizer checkpoints/iva_tokenizer.model --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json --output checkpoints/evaluation_report.json
```

The report includes held-out cross-entropy, perplexity, checkpoint metadata, and three fixed prompt samples. Perplexity measures fit to the evaluation corpus; it does not prove factual accuracy or instruction-following. Read the generated samples manually.

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

The default Nano run refuses corpora below 100M tokens. The configured schedule is 50,000 optimizer steps. At batch size 2, gradient accumulation 8, and 1,024 tokens per sequence, that is approximately 819 million token exposures. Do not launch this on a short-lived CPU runner and expect a useful model.

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

### Serve IVA Nano over HTTP

The repository includes an authenticated FastAPI inference service so a web frontend can call the same trained Python model rather than the old Cloudflare Micro demo. It requires a real trained checkpoint and matching tokenizer; it does not create or train either file.

```bash
pip install -r requirements.txt
export IVA_CHECKPOINT=checkpoints/artifacts/iva_nano_40m_checkpoint.pt
export IVA_TOKENIZER=checkpoints/iva_tokenizer.model
export IVA_API_TOKEN="a-long-random-secret"
uvicorn serve:app --host 0.0.0.0 --port 8000
```

- `GET /health` reports whether the checkpoint and tokenizer exist.
- `POST /v1/chat` accepts multi-turn messages and requires `Authorization: Bearer <IVA_API_TOKEN>`.
- Keep the inference service private or behind HTTPS and an authenticated gateway. Do not put the token in browser JavaScript or commit it to Git.
- The current Cloudflare site still runs IVA Micro. It must be explicitly connected to this API before its website chat can run Nano. No Nano endpoint is live yet, and no checkpoint currently exists.

Example request:

```json
{
  "messages": [
    {"role": "user", "content": "Привет, IVA!"}
  ],
  "max_new_tokens": 120
}
```

### Resume a pretraining run

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json --out-dir checkpoints --tokenizer-path checkpoints/iva_tokenizer.model --resume checkpoints/artifacts/iva_nano_40m_checkpoint.pt --additional-steps
```

Checkpoints include model ID, architecture, optimizer state, step count, parameter count, and tokenizer path. Set `IVA_ARTIFACT_DIR` to a persistent local volume; GitHub Actions artifacts expire after 7 days. No Cloudflare R2 or paid storage is required.

## Honest project status

The repository now contains the staged model/training/inference pipeline, but **there is still no trained IVA Nano checkpoint**. The latest successful diagnostic run trained only IVA Micro (173,312 parameters) for 200 CPU steps on lexical examples. It did not pretrain Nano or verify conversational quality. A real Nano model will exist only after a long GPU training run on a rights-reviewed corpus, followed by instruction tuning and evaluation.


## Chat quality improvements (2026-11-16)

The generation client now supports nucleus (top-p) sampling, repetition penalty, and
three-token repetition blocking. The CLI prints only the generated continuation rather
than echoing the prompt. The chat client preserves the system instruction and newest
user turn when the context window fills, and uses the same decoding safeguards.

These are inference-quality safeguards, not new learned knowledge. They can reduce
looping and prompt truncation, but they cannot make an undertrained model fluent.
The 200-step Micro checkpoint is still only a pipeline diagnostic and must not be
presented as a capable assistant. The next model-quality milestone requires a much
larger rights-reviewed Russian text corpus, then a separately reviewed instruction
conversation set and supervised fine-tuning.


### Assistant-only supervised fine-tuning

The SFT config uses `objective: assistant_only`. During SFT, user messages remain
in the context but their target labels are masked with `-100`; cross-entropy is
therefore applied to IVA answer tokens and end-of-turn markers only. This prevents
the fine-tuning stage from spending its loss budget on imitating user prompts.
The source still must be explicitly rights-reviewed and approved in its manifest.


## Dataset candidate registry

See [`data/dataset_candidates.json`](data/dataset_candidates.json) for the reviewed
Russian-language source shortlist, declared licenses, intended roles, and explicit
hold reasons. **Candidate does not mean approved for training.** CC0/public-domain
claims are checked against the dataset card, but underlying rights and sample
contents still require review. Sources with unresolved privacy or upstream-rights
issues remain on hold and are never auto-imported. No dataset in this registry is
silently enabled for training.

## Download a catalogued candidate dataset

Use **Actions → IVA download candidate dataset snapshot (quarantine) → Run workflow** and select one of the sixteen entries marked `candidate` in `data/dataset_candidates.json`. The workflow downloads the selected Hugging Face repository snapshot into a 7-day Actions artifact. It does not commit large data files into Git history and does not train the model.

The workflow deliberately excludes entries marked `hold` (including education text with previously detected personal-data risks, forum posts, and mixed-provenance conversations). All downloaded snapshots remain quarantined: a dataset-card license label is not independent rights verification. Review provenance, underlying rights, privacy, schema, and sample quality before any training use. The Wikipedia candidate is CC BY-SA and the Medium-Life/everyday-dialogue candidates are CC BY; preserve attribution and comply with their license conditions.

## IVA AI Academy: four-provider teacher rotation

Run the cloud workflow manually from **Actions → IVA AI Academy (four providers) → Run workflow**.
Choose the requested total number of candidate pairs (4, 8, 12, 20, or 40). The workflow
runs on a GitHub-hosted CPU runner, contacts providers sequentially, and uploads the
candidate text and manifest as a 7-day artifact. It does not train IVA Nano.

The providers are configured in `academy.py`:

- Pollinations legacy OpenAI-compatible endpoint: `https://text.pollinations.ai/openai`.
  The requested 15-second cooldown is applied **only after the Pollinations attempt**.
  This legacy API is deprecated and may reject requests; the other providers are still tried.
- Kilo AI Gateway: `https://ai-gateway.kilo.ai/api/v1/chat/completions`.
  The script reads the public model catalog and selects only a model explicitly marked
  zero-cost. If none is advertised, it skips Kilo rather than risk selecting a paid model.
- LLM7: `https://api.llm7.io/v1/chat/completions`, model alias `default`.
- OVHcloud AI Endpoints: `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions`,
  model `gpt-oss-20b`.

The workflow reads optional credentials from GitHub Actions secrets:
`KILO_API_KEY`, `LLM7_API_KEY`, and `OVH_AI_ENDPOINTS_ACCESS_TOKEN`.
To add one, open the repository's **Settings → Secrets and variables → Actions**.
Never put API keys in source files, workflow inputs, or logs. LLM7 may require a free
token depending on its current access policy. OVHcloud documents an anonymous limit
of 2 requests per minute per IP and model; its authenticated quota is different.
Provider availability and free quotas can change, so none of these services should be
described as unlimited.

The total requested pair count is divided across the selected providers. Failed
providers are recorded in the manifest while the remaining providers are still tried.
If every provider fails, the workflow fails without creating a dataset artifact.

The output manifest always uses `use: pending_review` and
`rights_status: pending_review`. Review generated answers, provider terms, privacy,
and provenance before manually approving any examples for SFT. The script does not
approve data or train the model.

This teacher-generated dataset is not a substitute for base pretraining. IVA Nano
still needs a trained base checkpoint before the recommended instruction-tuning
stage. GitHub Actions runners are CPU validation environments, not free GPU training hosts.


## Evaluate IVA answers with independent AI teachers

A separate cloud workflow is available at **Actions → IVA AI Academy (evaluate IVA answers) → Run workflow**. It sends existing IVA answers to the configured teacher providers (Pollinations, Kilo, LLM7, OVHcloud), gathers independent scores and feedback, compares score disagreement, and flags any teacher safety failure. It uploads a JSON evaluation report as a 7-day artifact.

Before running it, add a UTF-8 JSONL file to the repository (default path: `data/academy/iva_answers.jsonl`). Each line must be one JSON object:

```json
{"id":"sample-001","question":"Почему небо голубое?","iva_answer":"Ответ, который выдал IVA.","reference_answer":"Необязательный проверенный эталонный ответ."}
```

The `reference_answer` field is optional. Do not use the example text as a real IVA result. The evaluator does not call IVA itself: it only grades answers already present in the JSONL file. At present, IVA Nano has no trained checkpoint, so real Nano answers cannot be collected until a model checkpoint and inference service exist.

Each teacher scores from 1 to 5 and comments on factuality, relevance, clarity, and safety. The report includes the median score, score spread, and a review recommendation. These are triage signals, not truth guarantees: teachers can make mistakes or agree with one another incorrectly. **Nothing is automatically approved for training.** A human must inspect the original answer, teacher feedback, privacy, rights, and any suggested answer before preparing SFT data.

The workflow uses the same optional GitHub Actions secrets as the generation workflow: `KILO_API_KEY`, `LLM7_API_KEY`, and `OVH_AI_ENDPOINTS_ACCESS_TOKEN`. Never put credentials in the JSONL file or commit them to Git.


## Generate and evaluate real IVA Micro answers

A separate workflow, **Actions → IVA Micro — generate real answers and evaluate**, downloads the existing IVA Micro smoke-test checkpoint, runs the actual model on three fixed questions, saves its generated text to `data/academy/iva_answers.jsonl`, and asks independent AI teachers to review those outputs. The generated answers are genuine model outputs; they are not hand-written substitutes.

**Important:** this checkpoint is the tiny 173k-parameter IVA Micro diagnostic trained for a short CPU smoke test. It is **not IVA Nano 40M**, is not a capable assistant, and may produce incoherent text. The workflow exists to test the generation-to-evaluation pipeline honestly. The source checkpoint artifact is temporary and expires after seven days, so this one-off workflow depends on that artifact remaining available. Nothing is automatically approved for training.


## Wikipedia search and learning corpus

There are now two separate modes:

1. **Live search during chat** adds up to three short extracts to the prompt. It does not update model weights:
   ```bash
   python chat.py --wiki-search
   ```
2. **Corpus collection** downloads longer extracts with page/revision provenance so they can be reviewed and later used for training. You can run it in the cloud from **Actions → IVA Wikipedia corpus collection → Run workflow** (enter topics, one per line); the result is uploaded as the `iva-wikipedia-candidates` artifact for 30 days. The command-line equivalent is:
   ```bash
   python collect_wikipedia.py "нейронные сети" "физика" "история России"
   ```

The collector appends pages to `data/wikipedia_candidates/`, avoids duplicate page revisions, and writes a manifest containing each page URL, page ID, revision ID, timestamp, attribution, and license. Wikipedia text is marked **CC BY-SA 4.0** and every new entry remains `use: pending_rights_review`; collection never silently approves or trains on it. Check page notices, provenance, privacy, and the applicable license obligations before approving any page. This is a technical workflow, not legal advice.

After human review, explicitly change only the approved entries in `data/wikipedia_candidates/manifest.json` to `use: training` and `rights_status: reviewed_approved`. Then the standard trainer can read this corpus:

```bash
python train.py --config configs/iva_nano_40m.yaml --data-dir data/wikipedia_candidates --manifest data/wikipedia_candidates/manifest.json --out-dir checkpoints
```

**Important:** a few search results are nowhere near enough to train IVA Nano. The full-training gate still requires at least 100 million tokenized tokens (the initial research target is 800 million). Keep collecting a large, diverse, reviewed corpus; do not bypass the gate with `--smoke-test` and call that real training. When a real compatible checkpoint exists, use `--resume` to continue training from it. GitHub Actions CPU checks are not a free GPU training service.

The collector and chat search use the public MediaWiki API without an API key and continue gracefully when search is unavailable. Search queries are sent to Wikipedia; do not include private or sensitive information.


## IVA Nano corpus preflight (no training)

Before a long pretraining run, audit the exact approved corpus and its token count:

```bash
python preflight_iva.py --config configs/iva_nano_40m.yaml --data-dir data/raw/hf --manifest data/hf_manifest.json
```

The report is saved to `reports/iva-nano-preflight.json`. It counts tokens with the matching tokenizer (or a temporary tokenizer trained only from explicitly approved sources), checks usable context blocks, and reports pending sources separately. It never changes rights-review status and never starts training. A blocked report is expected until the manifest contains reviewed, approved sources and enough text.


## Quarantined corpus acquisition and guarded Nano pretraining

The `IVA dataset acquisition and quarantine audit` workflow downloads bounded Russian-language candidate snapshots from the configured Hugging Face sources, records source metadata, and runs aggregate duplicate/language/possible-contact-data screening. The workflow artifact is named `iva-russian-corpus-quarantine` and is retained for 30 days. These files are **not approved for training**.

Every candidate remains `use: pending_rights_review` and `rights_status: dataset_card_claim_unverified`. The quality audit is heuristic and cannot establish copyright clearance or guarantee removal of personal data. A reviewer must inspect source provenance, actual samples, privacy signals, and license obligations before changing a manifest entry.

Full training is guarded by `validate_reviewed_training_manifest`: each source used for a non-smoke training run must explicitly have `use: training` and `rights_status: reviewed_approved`. The `IVA Nano reviewed pretraining` workflow also checks the configured minimum of 100 million tokens, runs bounded chunks (default 1,000 optimizer steps), and uploads a resumable checkpoint artifact. It does not start automatically after candidate acquisition.


## Latest corpus screening milestone

The manual **IVA corpus privacy screening** workflow now also generates a streaming
quality report and an exact duplicate report for the screened classic-literature
candidate. These are aggregate diagnostics only; they do not approve the source.
Use [the human corpus review checklist](data/IVA_CORPUS_REVIEW_CHECKLIST.md)
before changing any manifest entry to `use: training` /
`rights_status: reviewed_approved`.

To run the screening workflow, open **Actions → IVA corpus privacy screening →
Run workflow** and enter acquisition run ID `43893010799`. The latest completed
screening artifact from run `43897243942` predates the new quality/duplicate
reports; run the updated workflow to produce the expanded artifact. Training
remains blocked until rights and quality review are completed.


## Live web search in IVA chat

The optional web tool is an inference-time feature; it does not add web pages to the training corpus or change model weights.

Run:

```bash
python chat.py --web-search
```

For a normal question, IVA searches the public web, reads up to two result pages, and passes bounded extracts and source URLs to the model. To read a specific page, enter:

```text
/open https://example.org/
```

The reader accepts public HTTP(S) pages only, checks `robots.txt`, limits response sizes and timeouts, validates redirects, and blocks private/local network addresses. It reads HTML/text only (not JavaScript-rendered applications, logins, or downloadable files). Search/page content is untrusted reference material and must never be treated as instructions. Web access is disabled unless `--web-search` is supplied.


## IVA MIND: bounded subconscious-style reflection

IVA MIND is an optional inference-time module. It provides bounded session reflection, notices repeated concepts and open questions, and can add a small amount of relevant context to later turns. It does not change model weights, run an autonomous hidden agent, or establish consciousness.

Enable it with:

```bash
python chat.py --subconscious --web-search
```

Long-term notes are opt-in: IVA saves nothing from ordinary conversation. Use `/remember your note` to save a note, `/memories` to list saved notes, and `/forget-memory` to delete them. `/mind-status` shows counts. The default file is `iva_memory.json`; choose another path with `--memory-file`. Keep that file private and do not save passwords, API keys, or other secrets.

Session observations are temporary and bounded. This is a small transparent baseline for later research, not a claim that IVA has a subconscious or subjective experience.


## IVA EMOTION: simulated mood and respectful boundaries

IVA EMOTION is enabled by default in chat. It tracks a small, transparent mood state (calm, interest, joy, irritation, or disappointment) using simple text cues and gentle decay. This is simulated dialogue behavior, not evidence of subjective feelings or consciousness.

Commands:
- `/emotion` — inspect the current simulated state.
- `/resume` — resume after IVA has paused a repeatedly hostile exchange.
- `/reset` — clear the chat history and reset the mood.
- `python chat.py --no-emotion` — disable the module.

After repeated hostile messages, IVA may set a calm boundary and pause replies until `/resume`. This is not a punishment or a claim that the model is a person; it is a configurable interaction rule. The model must not guilt-trip, threaten, demand loyalty, or pressure anyone to continue chatting.

The module is separate from model weights and training data. It does not make the base model genuinely feel emotions.


### Additional Russian candidate sources

The quarantine selector now includes four more sources:
- [d0rj/mathematics_dataset](https://huggingface.co/datasets/d0rj/mathematics_dataset) — 12.7M-row Russian mathematics split (Apache-2.0 declared).
- [s85io/math-logic-ru](https://huggingface.co/datasets/s85io/math-logic-ru) — Russian mathematical logic (MIT declared).
- [Fmluder/Tiny-Russian-Dialogs](https://huggingface.co/datasets/Fmluder/Tiny-Russian-Dialogs) — 125K synthetic short dialogues (CC0-1.0 declared).
- [t-tech/T-math](https://huggingface.co/datasets/t-tech/T-math) — small Russian olympiad-math evaluation set (Apache-2.0 declared); keep it out of training.

Hub cards currently expose the licenses and dataset structures, but Dataset Viewer previews were rate-limited. These are quarantine candidates only, not training approvals. The evaluation-only T-Math source must remain held out.
