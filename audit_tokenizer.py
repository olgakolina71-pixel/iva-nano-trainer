import argparse
import json
from pathlib import Path

import yaml

from iva_nano.hf_compat import audit_sentencepiece_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Verify that IVA Nano's SentencePiece tokenizer matches its model vocabulary."
    )
    parser.add_argument("--config", default="configs/iva_nano_40m.yaml")
    parser.add_argument("--tokenizer", default="checkpoints/iva_tokenizer.model")
    parser.add_argument("--output", default="reports/tokenizer-audit.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    expected = int(config["model"]["vocab_size"])
    report = audit_sentencepiece_tokenizer(args.tokenizer, expected)
    report["model_vocab_size"] = expected
    report["compatible"] = True

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
