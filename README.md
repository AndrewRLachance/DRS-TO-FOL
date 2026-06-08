# DRS to FOL

This project contains experimental Python utilities for parsing, representing, transforming, lifting, pretty-printing, and generating first-order logic formulas derived from DRS-style (Discourse Representation Structure) and PMB ([Parallel Meaning Bank](https://pmb.let.rug.nl/data.php)) SBN-style  (Sequence Box Notation) semantic representations.

The project currently has two related areas of focus:

1. Converting PMB Simplified Box Notation (SBN) into NLTK-compatible DRS and then into First-Order Logic (FOL).
2. Representing and transforming first-order logic formulas as Python AST objects.

The broad pipeline is:

```text
PMB drs.sbn
→ SBNGraph
→ NLTK DRS string
→ NLTK DrtExpression
→ FOL
→ project-specific FOL AST
→ lifting / abstraction / pretty-printing
```

The intent is to enable accurate, quick, [natural language understanding](https://en.wikipedia.org/wiki/Natural_language_understanding).

## Important Status Notice

> [!NOTE]
> This only applies to [SBN to FOL conversions](sbn_to_fol.py)

This project is experimental and intended only as a launching point for further experimental downstream tasks.

Most of the code and documentation in this repository was generated with assistance from ChatGPT, specifically **GPT-5.5 Thinking**.

The generated results have only been reviewed by the developer. They have **not** been independently validated through formal proof, comprehensive test coverage, peer review, comparison against official PMB tooling, theorem-prover equivalence checking, or any other external verification process w/ the exception of the [English PMB Gold Dataset](data/gold.json) which was validated by trained human annotators.

It's important to note there is ample tooling available elsewhere to develop and guarantee correctness of this project.

Do not assume correctness for production, academic, benchmark, or publication use without additional validation.

## Project Structure

Current package layout:

```text
drs-to-fol
├── data
│   ├── gold-fol-lifted.json
│   ├── gold-fol.json
│   └── gold.json
├── junk
│   ├── collect.js
│   └── main.py
├── logic
│   ├── __init__.py
│   ├── fol_ast.py
│   ├── lifting
│   │   ├── __init__.py
│   │   ├── lift.py
│   │   ├── matchpy_lift.py
│   │   ├── pretty.py
│   │   └── quantifier_lift.py
│   └── parsing
│       ├── __init__.py
│       ├── drs.py
│       └── parser.py
├── pmb_scripts
│   ├── README
│   ├── graph_base.py
│   ├── long.py
│   ├── penman_model.py
│   ├── sbn2penman.py
│   ├── sbn_smatch.py
│   ├── sbn_smatch_fine_grained.py
│   ├── sbn_spec.py
│   ├── smatch.py
│   ├── smatch_fromlists.py
│   └── utils.py
├── fol_lift.py
└── sbn_to_fol.py
```

Some paths may differ depending on local development state.

## Package Overview

## SBN / DRS / FOL Conversion

### `sbn_spec.py`

Defines the SBN vocabulary and token classes used by the parser.

Important contents include:

* `SBN_NODE_TYPE`

  * `SYNSET`
  * `CONSTANT`
  * `BOX`

* `SBN_EDGE_TYPE`

  * `ROLE`
  * `DRS_OPERATOR`
  * `BOX_CONNECT`
  * `BOX_BOX_CONNECT`
  * `SYN_BOX_CONNECT`

* `SBNSpec.ROLES`

* `SBNSpec.DRS_OPERATORS`

* `SBNSpec.NEW_BOX_INDICATORS`

* regex helpers for:

  * WordNet synset IDs
  * SBN role offsets
  * SBN scope markers
  * quoted names
  * constants

This file was originally based on an older PMB 4-era SBN spec. Several additions and corrections were required while testing against PMB 5.1 data.

Important PMB 5.1-related additions include:

```python
"COMMENTARY"
"CauserOf"
```

`COMMENTARY` should be treated as a box/scopal/discourse indicator.

`CauserOf` should be treated as a semantic role and may also be added to `INVERTIBLE_ROLES`.

### Recommended regex split for offsets and scope markers

Role offsets and scope markers should not be treated as the same thing.

Recommended patterns:

```python
INDEX_PATTERN = re.compile(r"^([-+<>]\d+)$")
ROLE_INDEX_PATTERN = re.compile(r"^[-+]\d+$")
SCOPE_INDEX_PATTERN = re.compile(r"^[<>]\d+$")
```

Examples:

```text
Agent -1        # role offset
Theme +2        # role offset
NEGATION <1     # scope marker
Proposition >1  # scope marker
CONTINUATION <0 # scope marker
```

Role offsets point to synset nodes.

Scope markers describe discourse or scopal structure.

### `graph_base.py`

Defines a small `networkx.DiGraph` wrapper used by `SBNGraph`.

It provides helpers for:

* DOT visualization
* PNG/PDF graph rendering
* node and edge label rendering
* format export through `pydot`

### `sbn2penman.py`

Contains the main `SBNGraph` parser.

Despite the filename, the useful part for this project is not only the PENMAN export. The parser is also used as the front end for SBN-to-FOL conversion.

Main responsibilities:

* parse multiline PMB-style SBN
* parse one-line SBN when `is_single_line=True`
* create graph nodes for:

  * boxes
  * synsets
  * constants
* create graph edges for:

  * box membership
  * semantic roles
  * DRS/scopal operators
  * box-to-box discourse/scopal relations
* resolve relative SBN role indices like:

  * `Agent -1`
  * `Theme +2`
  * `Location +3`

The parser originally exported SBN graphs to PENMAN notation. This project adds a separate path from `SBNGraph` to NLTK DRS/FOL.

### `sbn_to_fol.py`

Adds the experimental exporter:

```text
SBNGraph → NLTK DRS string → NLTK FOL
```

It uses:

```python
from nltk.sem.drt import DrtExpression
```

and converts generated DRS strings with:

```python
DrtExpression.fromstring(drs).fol()
```

This is currently the main SBN-to-FOL conversion layer.

## FOL AST / Parsing / Lifting

### `logic.fol_ast`

Defines the AST classes used to represent first-order logic formulas.

Expected exported classes include:

* `Formula`
* `Var`
* `Atom`
* `Negation`
* `Conjunction`
* `Disjunction`
* `Implication`
* `BiImplication`
* `Exists`
* `Forall`

The purpose of this layer is to avoid tying all downstream formula manipulation to NLTK, Z3, PySMT, or any other solver-specific representation.

### `logic.parsing`

Contains parsing utilities for converting textual FOL into AST objects.

Primary exports:

```python
from logic import parse_fol, parse_fol_list_text, parse_fol_list_file
```

### `logic.lifting`

Contains logic transformation utilities, including quantifier lifting and pretty-printing.

Primary exports:

```python
from logic import lift_formula, to_string
```

#### Scoring API

- `score_formula(formula) -> FormulaScore`
- `score_delta(before, after) -> int`
- `is_better_lift(before, after) -> bool`
- `best_formula(candidates) -> Formula`
- `lift_formula_with_scores(formula) -> LiftResult`

`FormulaScore` includes structural metrics such as node count, negation count, implication count, quantifier count, max depth, abstraction score, penalty, and total score.

The scoring heuristic prefers formulas with:

1. more abstract connectives, especially `→` and `↔`
2. universal quantifier abstractions when they replace negated existentials
3. fewer negations
4. fewer disjunctions
5. fewer total AST nodes
6. shallower trees

#### Main rewrite families

- `¬¬P -> P`
- `¬∃x.P -> ∀x.¬P`
- `¬∀x.P -> ∃x.¬P`, accepted only when scoring prefers it
- `¬∃x.(A1 ∧ ... ∧ ¬B) -> ∀x.((A1 ∧ ...) → B)`
- `¬P ∨ Q -> P → Q`
- `(P → Q) ∧ (Q → P) -> P ↔ Q`

Candidate rewrites are score-gated. A rewrite is kept only when `is_better_lift(before, after)` ranks the candidate higher than the prior formula for that pass.


## Usage

### Convert PMB SBN JSON to DRS/FOL

The current `sbn_to_fol.py` workflow expects a JSON file shaped like:
```json
[
  {
    "raw": "...",
    "sbn": "..."
  }
]
```

Expected output:
```json
[
  {
    "fol": "..."
    "drs": "...",
    "raw": "...",
    "sbn": "..."
  }
]
```

Example:

```bash
python sbn_to_fol.py \
  --input <project-parent-directory>/drs-to-fol/gold.json \
```

The script prints the source SBN, generated DRS, and generated FOL.

### Convert one-line SBN

If the input file contains one flattened SBN per line:

```bash
python sbn_to_fol.py \
  --input sbn_template.txt \
  --single-line \
```

### Example: Parsing and Lifting a Formula

```python
from logic import parse_fol, lift_formula, to_string

formula = parse_fol("exists x. car(x)")
lifted = lift_formula(formula)

print(to_string(lifted))
```

### CLI Script for Batch Lifting

A CLI utility can be used to read a JSON file containing FOL samples, lift each formula, and write a new JSON file with the lifted output.

Example usage:

```bash
python fol_lift.py \
  --input gold-fol.json \
  --output gold-fol-lifted.json
```

With explicit `PYTHONPATH`:

```bash
PYTHONPATH=<project-parent-directory>/drs-to-fol \
python fol_lift.py \
  --input gold-fol.json \
  --output gold-fol-lifted.json
```

## Input JSON Formats

## SBN input

The SBN converter expects a JSON list of objects.

Each object should include:

```json
[
  {
    "fol": "exists e2 t3 x1 x4 x5.(entity_n_01(x1) & be_v_02(e2) & time_n_08(t3) & male_n_02(x4) & nickname_n_01(x5) & EQU(x1,EMPTY) & Co_Theme(e2,x1) & Time(e2,t3) & Theme(e2,x5) & EQU(t3,now) & Name(x4,Frank_Sinatra) & Bearer(x5,x4))",
    "raw": "What is Frank Sinatra's nickname?\r\n",
    "lifted": "∃e2 t3 x1 x4 x5.(entity_n_01(x1) ∧ be_v_02(e2) ∧ time_n_08(t3) ∧ male_n_02(x4) ∧ nickname_n_01(x5) ∧ EQU(x1,EMPTY) ∧ Co_Theme(e2,x1) ∧ Time(e2,t3) ∧ Theme(e2,x5) ∧ EQU(t3,now) ∧ Name(x4,Frank_Sinatra) ∧ Bearer(x5,x4))"
  }
]
```

The `sbn` field is required for SBN-to-FOL conversion.

## FOL lifting input (same as SBN conversion output)

The batch lifting script expects a JSON list of objects.

Each object should include at least:

```json
[
  {
    "drs": "DRS([x1,e2,t3,x4,x5],[entity_n_01(x1), be_v_02(e2), time_n_08(t3), male_n_02(x4), nickname_n_01(x5), EQU(x1,EMPTY), Co_Theme(e2,x1), Time(e2,t3), Theme(e2,x5), EQU(t3,now), Name(x4,Frank_Sinatra), Bearer(x5,x4)])",
    "fol": "exists e2 t3 x1 x4 x5.(entity_n_01(x1) & be_v_02(e2) & time_n_08(t3) & male_n_02(x4) & nickname_n_01(x5) & EQU(x1,EMPTY) & Co_Theme(e2,x1) & Time(e2,t3) & Theme(e2,x5) & EQU(t3,now) & Name(x4,Frank_Sinatra) & Bearer(x5,x4))",
    "sbn": "\n\nentity.n.01   EQU ?                        \nbe.v.02       Co-Theme -1 Time +1 Theme +3 \ntime.n.08     EQU now                      \nmale.n.02     Name \"Frank Sinatra\"         \nnickname.n.01 Bearer -1                    \n",
    "raw": "What is Frank Sinatra's nickname?\r\n"
  }
]
```

The `fol` field is required.

The `raw` field is optional and is copied into the output if present.

## Output JSON Format

The generated lifting output is a JSON list of objects:

```json
[
  {
    "fol": "exists e2 t3 x1 x4 x5.(entity_n_01(x1) & be_v_02(e2) & time_n_08(t3) & male_n_02(x4) & nickname_n_01(x5) & EQU(x1,EMPTY) & Co_Theme(e2,x1) & Time(e2,t3) & Theme(e2,x5) & EQU(t3,now) & Name(x4,Frank_Sinatra) & Bearer(x5,x4))",
    "raw": "What is Frank Sinatra's nickname?\r\n",
    "lifted": "∃e2 t3 x1 x4 x5.(entity_n_01(x1) ∧ be_v_02(e2) ∧ time_n_08(t3) ∧ male_n_02(x4) ∧ nickname_n_01(x5) ∧ EQU(x1,EMPTY) ∧ Co_Theme(e2,x1) ∧ Time(e2,t3) ∧ Theme(e2,x5) ∧ EQU(t3,now) ∧ Name(x4,Frank_Sinatra) ∧ Bearer(x5,x4))"
  }
]
```

SBN-to-FOL output is currently printed to stdout unless separately wrapped in a batch writer.

## SBN / DRS Interpretation Notes

The project has been discussed in the context of translating PMB SBN-style semantic structures into first-order logic.

For example, a simplified SBN representation such as:

```text
country.n.02        Name "japan"
car.n.01            Source -1
steering_wheel.n.01 PartOf -1
be.v.03             Theme -1 Time +1 Location +2
time.n.08           EQU now
right.n.02
```

was interpreted as an existential DRS-like representation, roughly:

```text
There exists a country named Japan, a car sourced from Japan,
a steering wheel that is part of that car, and a current state
of that steering wheel being located on the right.
```

A corresponding FOL-style representation is:

```text
∃c ∃x ∃w ∃e ∃t ∃r (
    Country(c)
  ∧ Name(c, Japan)
  ∧ Car(x)
  ∧ Source(x, c)
  ∧ SteeringWheel(w)
  ∧ PartOf(w, x)
  ∧ Be(e)
  ∧ Theme(e, w)
  ∧ Time(e, t)
  ∧ t = now
  ∧ Location(e, r)
  ∧ RightSide(r)
)
```

This should not be read as a universal claim such as:

```text
For every Japanese car, its steering wheel is on the right.
```

A universal reading would require explicit universal quantification or an implication-style structure:

```text
∀x (
  Car(x) ∧ Source(x, Japan)
  →
  ∃w (
    SteeringWheel(w)
    ∧ PartOf(w, x)
    ∧ LocatedOnRight(w, now)
  )
)
```

## SBN-to-FOL Implementation Decisions

### 1. Use `SBNGraph` as the source of truth

Earlier experiments directly scanned SBN tokens and tried to infer role attachment. That worked for some examples but was too heuristic.

The current approach uses `SBNGraph.from_string()` first. This is better because the graph parser resolves offsets like:

```text
play.v.01 Agent -1 Theme +2
```

into actual graph edges.

### 2. Preserve WordNet sense identity

Synset predicates are emitted with sense information preserved.

Example:

```text
dog.n.01
```

becomes:

```text
dog_n_01(x)
```

rather than:

```text
dog(x)
```

This avoids collapsing distinct WordNet senses.

### 3. Use local DRS boxes

The exporter tries to respect box-local referent introduction.

Instead of hoisting all variables to the top-level DRS, each box introduces only the variables attached to synsets that are direct members of that box.

This matters for negation, implication, alternation, and other scoped structures.

### 4. Handle basic scopal operators

The converter currently handles:

```text
NEGATION
```

as:

```text
-(DRS(...))
```

It handles:

```text
CONDITION / PRECONDITION + CONSEQUENCE
```

as implication:

```text
(DRS(...) -> DRS(...))
```

It handles:

```text
ALTERNATION
```

as disjunction:

```text
(DRS(...) | DRS(...))
```

Other discourse operators are conservatively embedded.

### 5. Treat unsupported discourse operators conservatively

Operators such as:

```text
COMMENTARY
EXPLANATION
CONTRAST
ELABORATION
CONTINUATION
SOURCE
RESULT
CONJUNCTION
ATTRIBUTION
```

are not fully semantically interpreted yet.

For now, their child DRS content is generally preserved, while the discourse label itself may be dropped or treated structurally.

## Important Fixes Made During Development

### Multi-digit index support

The original index regex only handled one digit.

Recommended patterns:

```python
INDEX_PATTERN = re.compile(r"^([-+<>]\d+)$")
ROLE_INDEX_PATTERN = re.compile(r"^[-+]\d+$")
SCOPE_INDEX_PATTERN = re.compile(r"^[<>]\d+$")
```

This prevents `-12`, `<10`, and similar markers from being misread.

### Separate role offsets from scope markers

These are not the same:

```text
Agent -1
Theme +2
```

versus:

```text
NEGATION <1
Proposition >1
CONTINUATION <0
```

Role offsets point to synset nodes.

Scope markers describe scopal/discourse structure and should not be treated as ordinary synset references.

### Count only synset lines for valid synset offsets

Standalone operator lines such as:

```text
NEGATION <1
CONTINUATION <0
COMMENTARY <5
```

are not synset lines.

The parser should not use `len(lines) - 1` as the maximum synset index. It should count only lines whose first token is a synset.

### Add PMB 5.1 vocabulary

Additional PMB 5.1 items encountered include:

```text
COMMENTARY
CauserOf
```

`COMMENTARY` should be included in `NEW_BOX_INDICATORS`.

`CauserOf` should be included in `ROLES`, and optionally in `INVERTIBLE_ROLES`.

### Tolerate unknown PMB-style operators

Because the spec may lag behind PMB 5.1, the parser can optionally tolerate uppercase operator-like tokens followed by scope markers, for example:

```text
SOME_OPERATOR <3
```

These can be converted into box edges instead of crashing.

### Avoid phantom graph nodes

A bug occurred when expressions such as:

```text
Proposition <1
```

were interpreted as ordinary forward synset offsets.

That created graph edges to target nodes that did not actually exist, causing missing node attributes later.

The fix is to treat `<N` and `>N` as scope markers, not normal role offsets.

### Skip scope-marker pseudo-edges during FOL emission

Pseudo-edges like:

```text
Proposition <1
```

should not be emitted as binary FOL predicates like:

```text
Proposition(e,C_1)
```

They need separate scopal handling. Until that is implemented, the exporter skips these pseudo-edges.

## Development Caveats

Known areas requiring further review:

* Parser correctness across the full expected FOL grammar.
* Variable capture avoidance.
* Scoping behavior for negation, implication, conjunction, disjunction, and biconditional.
* Pretty-printer round-tripping.
* DRS/SBN-to-FOL semantic assumptions.
* Compatibility between `matchpy_lift.py`, `quantifier_lift.py`, and the primary lifting API.
* Test coverage for malformed input.
* Formal validation against gold data.
* Correctness of PMB SBN box/scopal reconstruction.
* Proposition scope handling.
* Modal semantics.
* Temporal relation semantics.
* Equality semantics.
* Quantity and numeric value handling.
* Discourse relation interpretation.

## Known Limitations

### Not officially validated

The converter has not been validated against official PMB DRS, Boxer output, gold clause notation, theorem-prover equivalence, or any external semantic benchmark.

Current validation is limited to developer review and runtime testing.

### Not full PMB semantics

The output is intended to be useful and parseable, but it is not guaranteed to preserve all PMB semantics.

Known weak areas:

* proposition scoping
* discourse relations
* modal operators
* attribution
* commentary
* continuation
* temporal operators
* equality semantics
* quantities
* numeric values
* generalized quantification
* accessibility constraints across DRS boxes

### Modal operators are not faithfully represented

NLTK DRT does not provide native modal FOL operators for:

```text
POSSIBILITY
NECESSITY
```

The current converter preserves the embedded content but does not faithfully encode modality.

### Discourse relations are mostly structural

Relations like:

```text
EXPLANATION
CONTRAST
COMMENTARY
ELABORATION
```

are not given full logical semantics.

### `Proposition <N` is not fully interpreted

Currently, scope-marker proposition edges are skipped to avoid producing invalid or misleading binary FOL predicates.

A more faithful version would represent proposition-taking predicates with embedded DRS arguments or a higher-order/semantic-frame representation.

### Generated FOL may be approximate

Passing `DrtExpression.fromstring(...).fol()` means the generated DRS is syntactically acceptable to NLTK. It does not mean the formula is semantically equivalent to the intended PMB representation.

## Recommended Validation Work

Before relying on the output, add tests for:

* Atomic predicates.
* Nested conjunctions and disjunctions.
* Negation scope.
* Implication and biconditional scope.
* Existential and universal quantifiers.
* Mixed quantifier nesting.
* Variable shadowing.
* Variable capture during lifting.
* Parser / pretty-printer round trips.
* Batch JSON processing.
* SBN role offset resolution.
* SBN scope marker handling.
* PMB 5.1-specific discourse operators.
* SBN-to-DRS-to-FOL smoke tests.
* Malformed SBN and malformed FOL input.

Example test categories:

```text
tests/
├── test_parser.py
├── test_fol_ast.py
├── test_lifting.py
├── test_pretty.py
├── test_cli.py
├── test_sbn_parser.py
├── test_sbn_to_drs.py
└── test_sbn_to_fol.py
```

## Recommended Next Steps

### 1. Add regression tests

Create a small test set of SBN snippets covering:

* flat role structures
* negation
* double negation
* conditionals
* alternation
* commentary
* continuation
* proposition scopes
* constants
* names
* numeric values
* multi-digit offsets
* unknown roles
* unknown discourse operators

### 2. Add parser diagnostics

Track:

* skipped scope-marker edges
* unknown operators
* unknown roles
* possibly ill-formed graphs
* constants that look like indices
* unsupported scopal structures

### 3. Add a neutral intermediate representation

A better long-term architecture is:

```text
SBNGraph
→ DRSBox IR
→ Formula IR
→ NLTK / Z3 / PySMT / TPTP / JSON
```

This avoids making NLTK the central semantic representation.

### 4. Build semantic abstraction layers

Useful abstractions include:

```text
Entity(x, synset="dog.n.01")
Event(e, synset="play.v.01", roles={Agent: x, Theme: y})
Time(t, relation="TPR", value="now")
Name(x, "China")
Quantity(x, 30)
```

This may be more useful than raw FOL for many downstream applications.

### 5. Compare against any available PMB tooling

If an official or community-supported PMB SBN-to-DRS or SBN-to-CLF converter is found, compare this converter against it.

### 6. Add output writers

Currently, some scripts print to stdout. Add JSON writers for:

* raw SBN
* generated DRS
* generated FOL
* parsed FOL AST
* lifted FOL
* diagnostics


## AI-Assisted Development Disclosure

This repository contains code and documentation generated primarily with assistance from ChatGPT, specifically **GPT-5.5 Thinking**.

The developer reviewed the generated outputs, but the results have not been independently validated by formal methods, third-party review, comprehensive automated testing, official PMB tooling comparison, or any other external correctness process.

## Citation

If you use this project, please cite:

Lasha Abzianidze, Johannes Bjerva, Kilian Evang, Hessel Haagsma, Rik van Noord, Pierre Ludmann, Duc-Duy Nguyen, and Johan Bos. 2017. [*The Parallel Meaning Bank: Towards a Multilingual Corpus of Translations Annotated with Compositional Meaning Representations*](https://aclanthology.org/E17-2039.pdf). In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL), pages 242–247, Valencia, Spain.