"""
Linear SBN -> NLTK DRT (DRS string) converter (hardened for NLTK parsing)

Goal:
  Produce DRS strings that parse with:
      from nltk.sem.drt import DrtExpression
      drs = DrtExpression.fromstring(drs_str)
      fol = drs.fol()

Key features implemented (per our debugging trail):
  1) Synset -> referent + unary predicate:
       noun.*      -> x#
       time.n.*    -> t#
       v/a/r.*     -> e#
     Adds unary predicates like lung_cancer(x1), find(e1), timely(e2).

  2) Role edges with +/- offsets:
       Agent -1, Theme +2, ...
     Emits Role(e_head, target_ref), attaching to nearest event referent on the left (e*),
     and adding the condition into the nearest predicate-frame to the left when possible.

  3) VALUE roles (Name, EQU, TPR, Quantity, ...):
     - Consumes a "value phrase" of 1+ tokens (handles things like: Prevention Editorial Board, 6 A, PGE(2)).
     - If first value token is +N/-N, treats it as a pointer to another referent (never emits +2 as a constant).
     - Sanitizes constants HARD so NLTK never sees:
         * spaces
         * parentheses: PGE(2) -> PGE_2
         * decimals as tokens: 8.8 -> N8_8
         * single-letter constants: i -> C_i
     - Never emits malformed Role(x,) calls.

  4) Proposition scoping (segment-based; stack-driven):
     - Proposition >k after a predicate: introducer expects k *segments*
     - Segment boundaries ONLY at: CONTINUATION <0
     - On boundary: close previous segment, activate pending propositions for next segment,
       attach closed segments to the active proposition introducer as embedded DRS.
     - CONTINUATION <k> where k>0 treated as back-link (ignored for segmentation for NLTK projection).

  5) Operator projection (frame-based heuristic; NLTK-parseable):
     - NEGATION <k> -> -(DRS([],[...]))
     - CONDITION/PRECONDITION <k> + CONSEQUENCE <k> -> (A -> B)
     - ALTERNATION <k> -> (A | B)
     - other unary discourse/modals with <k -> keep content as DRS([],[...]) and drop label

  6) Predicate-name hardening:
     - Avoid NLTK reserved tokens like 'exist', 'all', etc. by prefixing p_
     - Avoid var-like predicates and single-letter predicates: m(x) -> p_m(x), t(x)->p_t(x)
     - Avoid any predicate/constant whitespace & punctuation problems.

Usage:
  python sbn_to_nltk_drt.py --input /mnt/data/drs.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple, Set
from nltk.sem.drt import DrtExpression

dexpr = DrtExpression.fromstring

# ----------------------------
# Regex & tokenization
# ----------------------------

SENSE_RE = re.compile(r"^([A-Za-z0-9_-]+)\.([nvar])\.(\d+)$")
ROLE_OFFSET_RE = re.compile(r"^[+-]\d+$")
IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

INT_RE = re.compile(r"^-?\d+$")
FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
PLUS_FLOAT_RE = re.compile(r"^\+\d+\.\d+$")

# NLTK treats many single-letter / variable-like tokens specially. Avoid using these as predicates.
VARLIKE_TOKEN_RE = re.compile(r"^[a-z]\d*$")

NLTK_RESERVED = {
    # quantifiers / connectives / booleans / common keywords in NLTK logic grammar
    "all", "exists", "exist",
    "and", "or", "not",
    "true", "false",
    "iff", "implies",
}


def tokenize(s: str) -> List[str]:
    """Split on whitespace, preserving quoted strings as one token."""
    out: List[str] = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] == '"':
            j = i + 1
            while j < len(s) and s[j] != '"':
                j += 1
            out.append(s[i : j + 1])  # includes quotes
            i = j + 1
        else:
            j = i
            while j < len(s) and not s[j].isspace():
                j += 1
            out.append(s[i:j])
            i = j
    return out


def parse_synset(tok: str) -> Optional[Tuple[str, str, str]]:
    m = SENSE_RE.match(tok)
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3))


def is_synset(tok: str) -> bool:
    return parse_synset(tok) is not None


def is_predicate_synset(tok: str) -> bool:
    ps = parse_synset(tok)
    return ps is not None and ps[1] in {"v", "a", "r"}  # verb/adj/adv -> event-like


def norm_name(s: str) -> str:
    return s.replace("-", "_")


# ----------------------------
# NLTK-safe identifiers (predicates & constants)
# ----------------------------

def sanitize_ident(s: str) -> str:
    """
    Convert arbitrary string into an NLTK-safe identifier token:
      - replaces ANY non-alphanumeric with underscores
      - strips leading/trailing underscores
      - forces leading letter
      - avoids reserved tokens
      - avoids single-letter bare tokens (treat as variable-ish)
    """
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    if not s:
        return "EMPTY"
    if not re.match(r"^[A-Za-z]", s):
        s = "C_" + s
    if s.lower() in NLTK_RESERVED:
        s = "C_" + s
    if re.fullmatch(r"[A-Za-z]", s):
        s = "C_" + s
    return s


def float_to_const(tok: str) -> str:
    """Convert decimal token to safe identifier constant (no dot)."""
    if tok.startswith("+"):
        tok = tok[1:]
    neg = tok.startswith("-")
    if neg:
        tok = tok[1:]
    return sanitize_ident(("Nneg" if neg else "N") + tok.replace(".", "_"))


def drt_atom(tok: str) -> str:
    """
    Convert a token or synthesized value string into an NLTK-safe term:
      - quoted strings -> sanitize contents (no quotes in output)
      - integers -> keep as-is
      - decimals -> N8_8 style
      - anything else -> sanitize_ident (fixes PGE(2), spaces, punctuation, etc.)
    """
    if (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
        return sanitize_ident(tok[1:-1])

    if INT_RE.fullmatch(tok):
        return tok

    if FLOAT_RE.fullmatch(tok) or PLUS_FLOAT_RE.fullmatch(tok):
        return float_to_const(tok)

    return sanitize_ident(tok)


def safe_pred_name(name: str) -> str:
    """
    Predicate symbols must not collide with NLTK keywords or variable-like tokens.
    """
    name = norm_name(name)
    if name.lower() in NLTK_RESERVED:
        return "p_" + name
    if len(name) == 1:
        return "p_" + name
    if VARLIKE_TOKEN_RE.fullmatch(name):
        return "p_" + name
    return name


# ----------------------------
# Operators & value roles
# ----------------------------

OPS_UNARY = {"NEGATION", "POSSIBILITY", "NECESSITY", "CONTINUATION", "CONTRAST", "EXPLANATION"}
OPS_ANTECEDENT = {"CONDITION", "PRECONDITION"}
OPS_CONSEQUENT = {"CONSEQUENCE"}
OPS_BINARY_DISJ = {"ALTERNATION"}

VALUE_ROLES = {
    "Name",
    "EQU", "NEQ", "APX", "MOR",
    "TPR", "TSU", "ESU",
    "TIN", "TAB",
    "Quantity", "Value", "Order",
    "ClockTime", "DayOfMonth", "MonthOfYear", "YearOfCentury",
}


def looks_like_role_with_offset(tokens: List[str], idx: int) -> bool:
    # regex.match returns Match | None; coerce to bool for type-checkers
    return (
        idx + 1 < len(tokens)
        and bool(IDENT_RE.match(tokens[idx]))
        and bool(ROLE_OFFSET_RE.match(tokens[idx + 1]))
    )


def is_value_boundary(tokens: List[str], idx: int) -> bool:
    """
    A conservative boundary detector for multi-token value phrases:
      stop if we hit:
        - operator tokens
        - Proposition
        - <k / >k markers
        - a synset token
        - a Role +/-N pattern
        - another VALUE role token (start of new value)
    """
    tok = tokens[idx]
    if tok in OPS_UNARY or tok in OPS_ANTECEDENT or tok in OPS_CONSEQUENT or tok in OPS_BINARY_DISJ:
        return True
    if tok == "Proposition":
        return True
    if tok.startswith("<") or tok.startswith(">"):
        return True
    if is_synset(tok):
        return True
    if looks_like_role_with_offset(tokens, idx):
        return True
    if tok in VALUE_ROLES:
        return True
    return False


def value_phrase_to_const(parts: List[str]) -> str:
    # Remove wrapping quotes on each part and join with underscores (never spaces)
    joined = "_".join(p.strip('"').strip("'") for p in parts)
    return drt_atom(joined)


# ----------------------------
# Main conversion
# ----------------------------

def convert_one_linear_sbn(sbn: str, *, debug: bool = False) -> str:
    toks = tokenize(sbn)

    # 1) Create referents for synsets + base unary conditions
    ref_by_i: Dict[int, str] = {}
    refs: List[str] = []
    base_conds: List[str] = []

    xi = ei = ti = 0

    for i, tok in enumerate(toks):
        ps = parse_synset(tok)
        if not ps:
            continue
        lemma, pos, _sn = ps
        lemma_norm = norm_name(lemma)
        pred = safe_pred_name(lemma_norm)

        if pos == "n":
            if lemma == "time":
                ti += 1
                v = f"t{ti}"
            else:
                xi += 1
                v = f"x{xi}"
            refs.append(v)
            ref_by_i[i] = v
            base_conds.append(f"{pred}({v})")
        else:
            ei += 1
            v = f"e{ei}"
            refs.append(v)
            ref_by_i[i] = v
            base_conds.append(f"{pred}({v})")

    def nearest_event_left(tok_i: int) -> Optional[str]:
        j = tok_i - 1
        while j >= 0:
            r = ref_by_i.get(j)
            if r and r.startswith("e"):
                return r
            j -= 1
        return None

    def nearest_referent_left(tok_i: int) -> Optional[str]:
        j = tok_i - 1
        while j >= 0:
            r = ref_by_i.get(j)
            if r:
                return r
            j -= 1
        return None

    # 2) Build predicate frames for v/a/r synsets
    frames: List[List[str]] = []
    pred_i_to_frame_idx: Dict[int, int] = {}

    for i, tok in enumerate(toks):
        if not is_predicate_synset(tok):
            continue
        lemma, _pos, _sn = parse_synset(tok)  # type: ignore[misc]
        head = ref_by_i[i]  # e#
        pred = safe_pred_name(norm_name(lemma))
        fi = len(frames)
        pred_i_to_frame_idx[i] = fi
        frames.append([f"{pred}({head})"])

    # 3) Attach VALUE roles and role-offset edges
    i = 0
    while i < len(toks):
        t = toks[i]

        # VALUE roles: Role VALUEPHRASE (or Role +/-offset pointer)
        if t in VALUE_ROLES:
            head = nearest_referent_left(i)
            if head is None:
                i += 1
                continue

            if i + 1 >= len(toks):
                i += 1
                continue

            # Pointer case: immediate +/-N
            if ROLE_OFFSET_RE.match(toks[i + 1]):
                off = int(toks[i + 1])
                target = ref_by_i.get(i + off)
                if target is not None:
                    frames.append([f"{t}({head},{target})"])
                i += 2
                continue

            # Collect a multi-token value phrase
            j = i + 1
            parts: List[str] = []
            while j < len(toks) and not is_value_boundary(toks, j):
                parts.append(toks[j])
                j += 1

            # If next token was a boundary (no value), skip safely
            if not parts:
                i = j
                continue

            atom = value_phrase_to_const(parts)
            if atom:
                frames.append([f"{t}({head},{atom})"])

            i = j
            continue

        # Binary role edge: Role +/-N
        if IDENT_RE.match(t) and i + 1 < len(toks) and ROLE_OFFSET_RE.match(toks[i + 1]):
            role = safe_pred_name(norm_name(t))
            off = int(toks[i + 1])
            head_e = nearest_event_left(i)
            targ = ref_by_i.get(i + off)

            if head_e and targ:
                cond = f"{role}({head_e},{targ})"
                # attach to nearest predicate frame to the left if possible
                pj = i - 1
                while pj >= 0 and pj not in pred_i_to_frame_idx:
                    pj -= 1
                if pj in pred_i_to_frame_idx:
                    frames[pred_i_to_frame_idx[pj]].append(cond)
                else:
                    frames.append([cond])

            i += 2
            continue

        i += 1

    # 4) Proposition scoping (segment-based on CONTINUATION <0)
    consumed = apply_proposition_scoping_segment_stack(
        toks=toks,
        pred_i_to_frame_idx=pred_i_to_frame_idx,
        frames=frames,
        debug=debug,
    )

    # 5) Operator projection (frame-based; skipping consumed)
    projected = apply_operator_projection(
        toks=toks,
        frames=frames,
        consumed_frames=consumed,
    )

    conds = base_conds + projected
    return f"DRS([{','.join(refs)}],[{', '.join(conds)}])"


# ----------------------------
# Proposition scoping
# ----------------------------

def apply_proposition_scoping_segment_stack(
    *,
    toks: List[str],
    pred_i_to_frame_idx: Dict[int, int],
    frames: List[List[str]],
    debug: bool = False,
) -> Set[int]:
    """
    Proposition >k:
      - queue proposition introducer (nearest predicate to left)
      - k counts *segments*

    Segments:
      - delimited ONLY by CONTINUATION <0

    Attachment:
      - when a segment closes (at CONTINUATION <0 or EOF), attach closed segment to the
        currently active proposition introducer as embedded DRS([],[...]).
    """
    consumed: Set[int] = set()

    # Segment predicate frames by CONTINUATION <0 only
    segments: List[List[int]] = []
    cur: List[int] = []

    def push():
        nonlocal cur
        segments.append(cur)
        cur = []

    i = 0
    while i < len(toks):
        if i in pred_i_to_frame_idx:
            cur.append(pred_i_to_frame_idx[i])

        if toks[i] == "CONTINUATION" and i + 1 < len(toks) and toks[i + 1].startswith("<"):
            k = int(toks[i + 1][1:])
            if k == 0:
                push()
            else:
                if debug:
                    print(f"[debug] CONTINUATION <{k}: back-link (ignored for segmentation)")
            i += 2
            continue

        i += 1

    push()

    pending: List[Tuple[int, int]] = []  # (introducer_frame_idx, remaining_segments)
    active_stack: List[dict] = []        # {"introducer": int, "remaining": int, "active_from_seg": int}
    cur_seg_no = 0

    def attach(seg_no: int):
        if not active_stack:
            return
        if seg_no < 0 or seg_no >= len(segments):
            return
        seg_frames = segments[seg_no]
        if not seg_frames:
            return

        top = active_stack[-1]
        if seg_no < top["active_from_seg"]:
            return

        introducer = top["introducer"]
        embedded_conds: List[str] = []
        for fi in seg_frames:
            if fi in consumed:
                continue
            embedded_conds.extend(frames[fi])
            consumed.add(fi)

        if embedded_conds:
            frames[introducer].append(f"DRS([],[{', '.join(embedded_conds)}])")

        top["remaining"] -= 1
        if top["remaining"] <= 0:
            active_stack.pop()

    i = 0
    while i < len(toks):
        if toks[i] == "Proposition" and i + 1 < len(toks) and (toks[i + 1].startswith(">") or toks[i + 1].startswith("<")):
            scope = toks[i + 1]
            try:
                k = int(scope[1:])
            except ValueError:
                k = 0

            pj = i - 1
            while pj >= 0 and pj not in pred_i_to_frame_idx:
                pj -= 1

            if pj in pred_i_to_frame_idx and k > 0:
                pending.append((pred_i_to_frame_idx[pj], k))

            i += 2
            continue

        if toks[i] == "CONTINUATION" and i + 1 < len(toks) and toks[i + 1].startswith("<"):
            k = int(toks[i + 1][1:])
            if k == 0:
                attach(cur_seg_no)
                cur_seg_no += 1

                # activate pending for next segment
                for introducer_fi, kk in pending:
                    active_stack.append({"introducer": introducer_fi, "remaining": kk, "active_from_seg": cur_seg_no})
                pending.clear()

            i += 2
            continue

        i += 1

    # EOF closes last segment
    attach(cur_seg_no)

    if pending and debug:
        print(f"[debug] pending propositions without CONTINUATION <0 boundary: {pending}")

    return consumed


# ----------------------------
# Operator projection
# ----------------------------

def apply_operator_projection(
    *,
    toks: List[str],
    frames: List[List[str]],
    consumed_frames: Set[int],
) -> List[str]:
    """
    Frame-based heuristic: <k consumes k remaining frames (skipping consumed_frames).
    Produces NLTK-parseable expressions.
    """
    cursor = 0

    def take_k(k: int) -> str:
        nonlocal cursor
        picked: List[List[str]] = []
        while cursor < len(frames) and len(picked) < k:
            if cursor not in consumed_frames:
                picked.append(frames[cursor])
            cursor += 1
        conds: List[str] = []
        for fr in picked:
            conds.extend(fr)
        return f"DRS([],[{', '.join(conds)}])"

    out: List[str] = []
    pending_ant: Optional[str] = None

    i = 0
    while i < len(toks):
        t = toks[i]

        if t in OPS_UNARY and i + 1 < len(toks) and toks[i + 1].startswith("<"):
            k = int(toks[i + 1][1:])
            if k > 0:
                sub = take_k(k)
                if t == "NEGATION":
                    if sub != "DRS([],[])":
                        out.append(f"-({sub})")
                else:
                    if sub != "DRS([],[])":
                        out.append(sub)
            i += 2
            continue

        if t in OPS_ANTECEDENT and i + 1 < len(toks) and toks[i + 1].startswith("<"):
            k = int(toks[i + 1][1:])
            pending_ant = take_k(k) if k > 0 else "DRS([],[])"
            i += 2
            continue

        if t in OPS_CONSEQUENT and i + 1 < len(toks) and toks[i + 1].startswith("<"):
            k = int(toks[i + 1][1:])
            cons = take_k(k) if k > 0 else "DRS([],[])"
            if pending_ant is not None:
                out.append(f"({pending_ant} -> {cons})")
                pending_ant = None
            else:
                out.append(cons)
            i += 2
            continue

        if t in OPS_BINARY_DISJ and i + 1 < len(toks) and toks[i + 1].startswith("<"):
            k = int(toks[i + 1][1:])
            left = take_k(k) if k > 0 else "DRS([],[])"
            right = take_k(k) if k > 0 else "DRS([],[])"

            if left != "DRS([],[])" and right != "DRS([],[])":
                out.append(f"({left} | {right})")
            elif left != "DRS([],[])":
                out.append(left)
            elif right != "DRS([],[])":
                out.append(right)
            i += 2
            continue

        i += 1

    # Emit remaining frames (each as its own embedded DRS) skipping consumed frames
    while cursor < len(frames):
        if cursor not in consumed_frames:
            out.append(take_k(1))
        else:
            cursor += 1

    return out


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert linear SBN strings to NLTK-parseable DRT (DRS) strings.")
    ap.add_argument(
        "--input",
        type=str,
        default="/home/ai-developer/development/drs-to-fol/gold.json",
        help="Path to JSON file containing a list of linear SBN strings (default: /mnt/data/drs.json).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max number of items to convert (0 = all).")
    ap.add_argument("--debug", action="store_true", help="Print debug info.")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of linear SBN strings.")

    n = len(data) if args.limit <= 0 else min(len(data), args.limit)

    for idx in range(n):
        sbn = data[idx]["sbn"]
        if not isinstance(sbn, str):
            raise ValueError(f"Item {idx} is not a string.")
        res = convert_one_linear_sbn(sbn, debug=args.debug)
        
        print(dexpr(res).fol())


if __name__ == "__main__":
    main()