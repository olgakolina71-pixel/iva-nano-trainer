# RusLang-Edu-100 — candidate review (2026-11-19)

## Decision

**Keep as a conditional candidate. Do not promote the full dataset into training yet.**

Source: https://huggingface.co/datasets/DatasetsEval/RusLang-Edu-100

## Structure inspected

- 100 records, Russian language.
- Hub exposes one physical split named `train`.
- The row-level `split` field contains **90 train records and 10 validation records**.
- Validation IDs: `rl-000010`, `rl-000020`, `rl-000030`, `rl-000040`, `rl-000050`, `rl-000060`, `rl-000070`, `rl-000080`, `rl-000090`, `rl-000100`.
- Metadata declares CC BY 4.0, expert-curated, high certainty, unique gold answers and verified review status. These are publisher-supplied claims, not independent proof.

## Problems found in targeted answer review

- `rl-000088`: the input asks to make a message more formal and says the report is needed tomorrow. The output invents a specific deadline (“до конца текущего рабочего дня”) and an “утреннее совещание” that were not supplied.
- `rl-000089`: the output turns “она просто огонь” into “Он отличного качества”, asserting a factual quality judgment not present in the input.

Both records are row-level `train` and must be excluded or corrected before training.

The remaining 88 row-level train records have not all received independent linguistic verification. A full 100-row structural inspection is not the same as proof that every grammar explanation is correct.

## Recommended handling

1. Preserve the ten validation records strictly for evaluation. Never train on them, even though the Hub physical split is called `train`.
2. Exclude or correct `rl-000088` and `rl-000089`.
3. Review every remaining `answer_short`, `output`, and `explanation` against reliable Russian grammar references.
4. Verify source provenance and uploader authority for the declared CC BY 4.0 license; preserve attribution.
5. If it passes, use only as a small Russian grammar/style SFT supplement, not as general pretraining.

## Training gate

- Candidate status unchanged.
- No manifest entry added with `use: training`.
- `training_enabled` remains `false`.
- No training job was started.
