import json
import subprocess
import sys
from pathlib import Path

import logic
import logic.lifting as lifting
from logic import parse_fol, to_string


def test_lifting_public_imports_are_available_from_logic_and_lifting():
    assert logic.lift_formula is lifting.lift_formula
    assert logic.lift_formula_with_scores is lifting.lift_formula_with_scores
    assert logic.LiftResult is lifting.LiftResult
    assert logic.FormulaScore is lifting.FormulaScore
    assert logic.score_formula is lifting.score_formula
    assert logic.score_delta is lifting.score_delta
    assert logic.is_better_lift is lifting.is_better_lift
    assert logic.best_formula is lifting.best_formula


def test_lift_formula_matches_score_aware_result():
    formula = parse_fol("-dog_n_01(x1) | bark_v_01(x1)")

    lifted = logic.lift_formula(formula)
    result = logic.lift_formula_with_scores(formula)

    assert lifted == result.lifted


def test_lift_result_delta_matches_total_score_difference():
    formula = parse_fol("-dog_n_01(x1) | bark_v_01(x1)")
    result = logic.lift_formula_with_scores(formula)

    assert result.delta == result.lifted_score.total - result.original_score.total
    assert result.delta == logic.score_delta(result.original, result.lifted)


def test_lift_result_not_improved_when_no_rewrite_is_accepted():
    formula = parse_fol("dog_n_01(x1)")
    result = logic.lift_formula_with_scores(formula)

    assert result.lifted == formula
    assert result.delta == 0
    assert result.improved is False


def test_known_lift_case_produces_improved_score():
    formula = parse_fol("-exists x1.(dog_n_01(x1) & -bark_v_01(x1))")
    result = logic.lift_formula_with_scores(formula)

    assert result.improved is True
    assert result.delta > 0
    assert to_string(result.lifted) == "\u2200x1.(dog_n_01(x1) \u2192 bark_v_01(x1))"


def test_fol_lift_cli_include_scores(tmp_path):
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "fol": "-exists x1.(dog_n_01(x1) & -bark_v_01(x1))",
                    "raw": "fixture",
                }
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(root / "fol_lift.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--include-scores",
        ],
        cwd=root,
        check=True,
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert output[0]["fol"] == "-exists x1.(dog_n_01(x1) & -bark_v_01(x1))"
    assert output[0]["raw"] == "fixture"
    assert output[0]["lifted"] == "\u2200x1.(dog_n_01(x1) \u2192 bark_v_01(x1))"
    assert output[0]["score"]["total"] < output[0]["lifted_score"]["total"]
    assert output[0]["score_delta"] == (
        output[0]["lifted_score"]["total"] - output[0]["score"]["total"]
    )
    assert output[0]["improved"] is True
