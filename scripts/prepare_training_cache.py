"""Build a disk-backed token cache from approved UTF-8 corpus files."""
import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="append", required=True, help="Input text file; may be repeated")
    parser.add_argument("--tokenizer", default="checkpoints/iva_tokenizer.model")
    parser.add_argument("--tokens", default="data/cache/iva_tokens.u16")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--report", default="reports/token-cache.json")
    args = parser.parse_args()

    text_paths = [Path(p) for p in args.text]
    for path in text_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizer_path = Path(args.tokenizer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if not tokenizer_path.exists():
        spm.SentencePieceTrainer.train(
            input=",".join(str(p) for p in text_paths),
            model_prefix=str(tokenizer_path.with_suffix("")),
            vocab_size=args.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            input_sentence_size=300000,
            shuffle_input_sentence=True,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            hard_vocab_limit=False,
        )
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if processor.get_piece_size() > 65535:
        raise ValueError("Tokenizer vocabulary does not fit the uint16 token cache")

    token_path = Path(args.tokens)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    lines_seen = 0
    empty_lines = 0
    nul_chars_removed = 0
    replacement_chars_seen = 0
    with token_path.open("wb") as output:
        for source in text_paths:
            with source.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    lines_seen += 1
                    nul_chars_removed += line.count(chr(0))
                    replacement_chars_seen += line.count(chr(0xfffd))
                    line = line.replace(chr(0), "").strip()
                    if not line:
                        empty_lines += 1
                        continue
                    ids = processor.encode(line, out_type=int)
                    ids.append(processor.eos_id())
                    arr = np.asarray(ids, dtype=np.uint16)
                    output.write(arr.tobytes())
                    total_tokens += int(arr.size)

    report = {
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab_size": processor.get_piece_size(),
        "token_file": str(token_path),
        "token_dtype": "uint16",
        "token_count": total_tokens,
        "source_files": [str(p) for p in text_paths],
        "lines_seen": lines_seen,
        "empty_lines_skipped": empty_lines,
        "nul_characters_removed": nul_chars_removed,
        "replacement_characters_preserved": replacement_chars_seen,
        "streaming": True,
        "training_approval_changed": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
