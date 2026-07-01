#!/usr/bin/env python3

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from logic import lift_formula, lift_formula_with_scores, parse_fol, to_string


def lift_fol_file(
    input_path: Path,
    output_path: Path,
    indent: int = 4,
    *,
    include_scores: bool = False,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    data = []

    for i, src in enumerate(samples):
        if "fol" not in src:
            raise KeyError(f"Sample at index {i} is missing required key: 'fol'")

        formula = parse_fol(src["fol"])

        if include_scores:
            result = lift_formula_with_scores(formula)
            item = {
                "fol": src["fol"],
                "raw": src.get("raw"),
                "lifted": to_string(result.lifted),
                "score": asdict(result.original_score),
                "lifted_score": asdict(result.lifted_score),
                "score_delta": result.delta,
                "improved": result.improved,
            }
        else:
            lifted = lift_formula(formula)
            item = {
                "fol": src["fol"],
                "raw": src.get("raw"),
                "lifted": to_string(lifted),
            }

        data.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse FOL formulas from a JSON file, lift quantifiers, and write lifted output."
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input JSON file, e.g. gold-fol.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output JSON file, e.g. gold-fol-lifted.json",
    )

    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation level. Default: 4",
    )

    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="Include original/lifted score metadata in each output item.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lift_fol_file(
        args.input,
        args.output,
        args.indent,
        include_scores=args.include_scores,
    )


if __name__ == "__main__":
    main()
