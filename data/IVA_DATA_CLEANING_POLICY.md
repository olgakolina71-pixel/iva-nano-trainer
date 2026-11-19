# IVA dataset cleaning policy

Status: policy and pipeline requirements. No raw candidate corpus is modified by this document.

## Goal
Produce a reviewed, privacy-screened, deduplicated corpus from the candidate sources. The pipeline must be reproducible and must never silently convert a suspicious record into a trusted training example.

## Processing order
1. Fetch immutable source revisions and record source ID, revision, file path, source split, declared license, and SHA-256.
2. Parse only data files (JSON/JSONL/CSV/Parquet/plain text). Never execute dataset-provided Python loaders or scripts.
3. Normalize Unicode (NFC), line endings, and whitespace conservatively. Preserve code, punctuation, capitalization, mathematical notation, and message boundaries.
4. Remove empty/corrupt rows and obvious boilerplate; flag uncertain rows rather than deleting them automatically.
5. Detect personal data:
   - deterministic patterns for email addresses, phone numbers, URLs with credentials, IP addresses, payment/account-like identifiers, and common identity-number patterns;
   - context-based flags for names paired with addresses, school/class details, private correspondence, or other identifying combinations;
   - redact only when safe and unambiguous; otherwise quarantine the entire record.
   - never put raw matched values or raw examples in reports/logs.
6. Detect duplicates globally across all candidate sources:
   - exact normalized SHA-256;
   - near-duplicate text using MinHash/character or word shingles with language-aware thresholds;
   - conversation-level and message-level checks for SFT;
   - preserve one canonical record and keep a private provenance map; do not merge unrelated examples.
7. Detect low-quality or suspicious content:
   - malformed structure, encoding noise, repeated boilerplate, excessive repetition, language mismatch, empty/irrelevant answers, template spam, contradictions, and unsupported specific claims;
   - automated detectors only flag candidates. They do not establish factual truth.
8. Factuality review:
   - deterministic validation for math/code where executable tests are safe and sandboxed;
   - compare answer to prompt/context and use independent reference evidence for factual QA;
   - quarantine unverifiable, conflicting, or likely fabricated answers;
   - never invent replacement answers during cleaning.
9. Preserve original train/validation/test/reserve splits. Validation/test/benchmark data must never enter training. Deduplicate across splits and quarantine cross-split collisions to prevent leakage.
10. Export separate clean pretraining and SFT corpora, plus quarantine indexes and aggregate-only reports. Keep raw sources immutable.

## Required output artifacts
- `reports/cleaning_summary.json`: counts only (seen, kept, quarantined by reason, duplicates, redactions, language distribution, token estimates).
- `reports/source_provenance.json`: source IDs, pinned revisions, file hashes, split counts, license declarations and review status.
- `data/staging/pretrain/`: approved-to-review text records.
- `data/staging/sft/`: normalized message records.
- `data/quarantine/`: restricted artifacts, never included in training.
- `data/approved_manifest.json`: created only after maintainer review; never auto-generated as approved.

## Safety and correctness invariants
- No raw PII in logs, reports, Git commits, or public artifacts.
- A regex match is a flag, not proof; false positives require review.
- "Fake" is not reliably detectable by a single classifier. Mark factuality as verified, contradicted, unverifiable, or not-applicable with evidence; do not claim zero hallucinations.
- Synthetic content is not automatically true or high quality.
- Preserve source attribution and license/provenance metadata. Cleaning does not grant rights or remove obligations.
- Never set `training_enabled=true`, `use=training`, or `rights_status=reviewed_approved` as a side effect of cleaning.
- Training workflows must consume only the final reviewed manifest, never candidate/staging/quarantine directories.

## Acceptance criteria
A corpus can be proposed for maintainer approval only when:
- every retained row has source/revision/split provenance;
- PII scan reports zero unresolved high-confidence hits in the retained set;
- exact and near-duplicate passes have completed globally across all sources;
- all split collisions are quarantined;
- factuality/quality samples are manually reviewed per source and category;
- all questionable records are excluded or explicitly retained with a documented reason;
- aggregate reports contain no personal data or raw content.
