"""Validate source-specific normalization policies against the 29-entry IVA registry.

This checks catalog completeness and safety invariants. It does not download data,
change registry statuses, approve licenses, or add sources to a training manifest.
"""
import argparse
import json
from pathlib import Path

ALLOWED_DESTINATIONS = {
    "pretraining_candidate", "sft_candidate", "excluded_pending_rework",
    "train_small_candidate_plus_eval", "evaluation_or_english_only",
    "english_math_candidate", "english_physics_candidate",
    "blocked_pending_review", "blocked_pending_legal_review",
    "blocked_pending_rework", "blocked_pending_rights",
    "blocked_pending_provenance", "blocked_pending_license",
    "blocked_pending_row_license", "blocked_pending_schema_review",
    "blocked_pending_proof_validation", "blocked_noncommercial",
    "math_train_and_eval", "evaluation_only",
}


def validate(registry_path, rules_path):
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    registry_ids = [item["id"] for item in registry.get("datasets", [])]
    rule_items = rules.get("rules", [])
    rule_ids = [item["dataset_id"] for item in rule_items]

    errors = []
    if len(registry_ids) != 29:
        errors.append(f"Expected 29 registry entries, found {len(registry_ids)}")
    if len(rule_ids) != len(set(rule_ids)):
        errors.append("Duplicate dataset_id in normalization rules")
    missing = sorted(set(registry_ids) - set(rule_ids))
    unknown = sorted(set(rule_ids) - set(registry_ids))
    if missing:
        errors.append("Missing rules: " + ", ".join(missing))
    if unknown:
        errors.append("Rules reference unknown datasets: " + ", ".join(unknown))

    for item in rule_items:
        dataset_id = item.get("dataset_id", "<missing>")
        if item.get("destination") not in ALLOWED_DESTINATIONS:
            errors.append(f"{dataset_id}: invalid destination {item.get('destination')!r}")
        for field in ("format", "transform", "required_checks", "reason"):
            if not item.get(field):
                errors.append(f"{dataset_id}: missing {field}")
        if item.get("destination") == "evaluation_only" and "never_train" not in " ".join(item.get("required_checks", [])) and dataset_id == "ttech_tmath_ru_eval":
            errors.append("T-math must be explicitly marked never_train")

    if registry.get("policy", {}).get("training_enabled") is not False:
        errors.append("Registry training_enabled must remain false until explicit approval")

    if errors:
        raise SystemExit("\n".join("ERROR: " + error for error in errors))
    print(f"OK: {len(registry_ids)} registry entries have exactly one normalization policy.")
    print("No license approval or training status was changed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="data/dataset_candidates.json")
    parser.add_argument("--rules", default="data/dataset_normalization_rules.json")
    args = parser.parse_args()
    validate(args.registry, args.rules)


if __name__ == "__main__":
    main()
