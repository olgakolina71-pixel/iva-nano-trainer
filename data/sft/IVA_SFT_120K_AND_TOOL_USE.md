# IVA SFT: target 120,000 Russian dialogues + tool-use curriculum

## Goal
Build a real supervised fine-tuning corpus of **120,000 unique Russian dialogues**. This is a target, not a claim that the data already exists. Do not fill the quota with repeated templates or fabricated conversations.

## Composition targets
- 80,000 general conversational examples: greetings, follow-up questions, explanations, summarization, coding help, uncertainty, and multi-turn context.
- 20,000 tool-use examples: choosing a tool, emitting a valid call, reading the tool result, and explaining the result.
- 20,000 domain/task examples: math, science, education, programming, planning, and factual Q&A.
- Keep separate train/validation/test groups by source and conversation family; never split near-duplicates across them.
- Tool-use target is at least 20,000 dialogues. The builder enforces this minimum.

## Tool protocol IVA must learn
The assistant emits a structured call, not prose pretending a tool was run:

`<tool_call>{"name":"calculator","arguments":{"expression":"18*24"}}</tool_call>`

The runtime executes only the existing allow-list:
- `calculator`
- `current_time`
- `web_search`

The tool result is returned as a labelled observation:
`<tool_result name="calculator">432</tool_result>`

IVA then writes a normal user-facing answer. A tool call is not itself a final answer. The model must also learn when **not** to call a tool, how to recover from empty/error results, and never to claim execution if no tool result was received.

## Input data contract
Place reviewed JSONL files in `data/sft/incoming/*.jsonl`. Each line is one conversation object:
- `id`: stable source identifier
- `language`: `ru`
- `messages`: ordered messages with roles `user`, `assistant`, or `tool`
- Tool-call assistant messages use `tool_call: {name, arguments}`
- Tool results use role `tool` and matching `name`

Example:
```json
{"id":"demo-001","language":"ru","messages":[{"role":"user","content":"Сколько будет 18*24?"},{"role":"assistant","tool_call":{"name":"calculator","arguments":{"expression":"18*24"}}},{"role":"tool","name":"calculator","content":"432"},{"role":"assistant","content":"18 × 24 = 432."}]}
```

## Build
Run:
```bash
python scripts/build_iva_sft_corpus.py --target-dialogues 120000 --minimum-tool-dialogues 20000
```
The builder rejects malformed records, duplicate conversations, unsupported tools, possible email/phone patterns, and non-Russian records. It writes nothing unless both quotas are met. It never downloads datasets or grants training approval.

## Current reality / gates
- The repository currently contains only 191 original dialogue pairs, not 120,000.
- The SFT training code requires an explicitly reviewed manifest and a base checkpoint.
- The dataset manifest remains disabled until source rights, privacy, quality, and split review are complete.
- A successful corpus build is not the same as a successful model training run.
- After building, run dataset-level quality review, then SFT, then held-out tool-use evaluations (correct tool choice, valid JSON arguments, correct use of tool results, abstention/no-tool cases, and error recovery).
