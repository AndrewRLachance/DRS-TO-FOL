# TODO Backlog

This backlog is derived from the README review. It separates quick documentation
fixes from validation work and larger implementation follow-ups.

## Documentation

- [x] Fix README examples so SBN-to-FOL output JSON is valid and sample paths
  point at `data/`.
- [x] Update the README project structure tree, including new modules such as
  `logic/lifting/scoring.py`.
- [x] Clarify canonical sample data locations under `data/` and document any
  root-level JSON files as local/generated artifacts.
- [x] Document all available output modes for `sbn_to_fol.py`, including
  `--output`, compact stdout JSON, and `--pretty`.
- [x] Keep README examples synchronized with the current CLIs and JSON schemas.
- [ ] Document output writer coverage for generated DRS, generated FOL, lifted
  FOL, parsed AST data, and diagnostics.

## Regression Tests

- [x] Add initial regression tests for SBN scope-marker handling, multi-digit
  role offsets, and basic SBN-to-DRS/FOL conversion.
- [ ] Add SBN parser tests for flat role structures, relative role offsets,
  multi-digit offsets, scope markers, names, constants, numeric values, unknown
  roles, PMB 5.1 vocabulary, and unknown discourse operators.
- [ ] Add SBN malformed-input tests covering invalid offsets, unsupported scope
  shapes, ill-formed graphs, and constants that look like indices.
- [ ] Add SBN-to-DRS and SBN-to-FOL smoke tests for negation, double negation,
  conditionals, alternation, commentary, continuation, and proposition scopes.
- [ ] Add FOL parser tests for precedence, nested conjunction/disjunction,
  existential and universal quantifiers, mixed quantifier nesting, malformed
  FOL, and variable shadowing.
- [ ] Add lifting and pretty-printer tests for variable capture avoidance,
  rewrite scoring, round trips, and batch JSON processing.
- [ ] Add CLI workflow tests for `sbn_to_fol.py` and `fol_lift.py`.
- [ ] Add corpus smoke tests over `data/gold.json`, `data/gold-fol.json`, and
  `data/gold-fol-lifted.json`.

## Diagnostics And Validation

- [x] Add parser/export diagnostics for skipped scope-marker edges, unknown
  roles, unknown operators, malformed graphs, constants that look like indices,
  and unsupported scopal structures.
- [ ] Add corpus-level validation reports for generated DRS/FOL success,
  conversion failures, skipped structures, and diagnostic frequencies.
- [ ] Compare generated output against official or community PMB tooling where
  available.
- [ ] Add theorem-prover or equivalence checks where they are practical for
  small focused examples.

## Semantic Modeling

- [x] Explicitly model or document the chosen behavior for `Proposition <N`.
- [x] Explicitly model or document modal semantics for `POSSIBILITY` and
  `NECESSITY`.
- [x] Explicitly model or document discourse relation semantics for attribution,
  commentary, continuation, contrast, elaboration, explanation, result, and
  source.
- [x] Review temporal relation semantics, equality semantics, quantity handling,
  numeric value handling, generalized quantification, and DRS accessibility
  constraints.
- [x] Add a neutral intermediate representation between `SBNGraph` and
  downstream formats so NLTK is not the central semantic representation.
- [x] Build optional semantic abstraction layers such as `Entity`, `Event`,
  `Time`, `Name`, and `Quantity`.

## Lifting Architecture

- [x] Reconcile `matchpy_lift.py`, `quantifier_lift.py`, scoring, and the public
  `lift_formula` API.
- [x] Document which lifting implementation is canonical.
- [x] Expose score-aware lifting results consistently if they remain part of
  the public API.
