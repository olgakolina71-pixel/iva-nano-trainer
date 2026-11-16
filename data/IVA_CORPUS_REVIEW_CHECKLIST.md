# IVA Nano corpus review checklist

This checklist is a human decision gate. Automated reports are evidence for review,
not an approval mechanism.

## Candidate under review

- Source: `Imperius/ru-classic`
- Snapshot artifact: acquisition run `43893010799`
- Screened candidate artifact: screening run `43902555939`
- Candidate artifact name: `iva-classic-screened-candidate-pending-review`
- Declared license: CC0 on the dataset card; this declaration is not independent
  proof that every underlying work and edition is free of third-party rights.

## Required checks before training

1. **Provenance and rights**
   - Read the current dataset card and repository history.
   - Identify the original editions/scans/transcriptions and their publication dates.
   - Check whether each underlying work is public domain in the relevant jurisdictions.
   - Check whether modern introductions, annotations, editorial notes, translations,
     OCR corrections, and source-specific metadata carry separate rights.
   - Record the reviewer, review date, evidence URLs, scope, and any exclusions.
   - If any material remains uncertain, keep the source pending or exclude it.

2. **Privacy**
   - Review the aggregate screening report and a controlled sample in a secure
     environment; do not publish raw excerpts that could expose personal data.
   - Heuristic phone/email matches may be false positives and false negatives.
   - Confirm that no personal contact details, private correspondence, or
     identifying information about living people are retained.

3. **Content and quality**
   - Review samples from the beginning, middle, and end of the source.
   - Check OCR noise, encoding, truncation, malformed fragments, metadata leakage,
     language mix, and repeated boilerplate.
   - Decide whether the line-based record format is appropriate for tokenizer and
     next-token pretraining.
   - Keep duplicate counts as evidence; do not blindly remove repeated lines when
     they may be legitimate fragments or structural markers.

4. **Decision**
   - Only a maintainer may update the approved manifest after all checks pass.
   - Record the exact source revision/hash, license, attribution, scope, and decision.
   - Set `use: training` and `rights_status: reviewed_approved` only for the
     reviewed source and only after documenting the evidence.
   - Otherwise retain `pending_rights_review` or mark the source excluded.

## Current status

**NOT APPROVED FOR TRAINING.** The privacy screening and quality/duplicate audit
workflows produce candidate reports only. They do not verify ownership, grant
permissions, or approve the corpus. The screened candidate contains Tolstoy's
`Война и мир` (including a record labelled `Первый вариант романа`), but the
same record's editorial introduction describes modern editorial sources. This
confirms presence, not clearance of the complete edition or corpus. No IVA Nano
training checkpoint is created by these workflows.

## Token-volume preflight (candidate only)

A manual GitHub Actions workflow, **IVA candidate token preflight (no training)**,
can estimate token volume for the screened artifact without changing its review
status:

- Workflow: `.github/workflows/iva-candidate-token-preflight.yml`
- Script: `scripts/candidate_token_preflight.py`
- Default source run: `43902555939`
- Default artifact: `iva-classic-screened-candidate-pending-review`

The report is explicitly an estimate using a sampled 24k SentencePiece tokenizer.
It is not the final approved-corpus preflight and does not authorize training.
The workflow does not create model weights or alter any manifest.
