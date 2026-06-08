#!/usr/bin/env python3

from __future__ import annotations

import json
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from nltk.sem.drt import DrtExpression

from pmb_scripts.sbn2penman import SBNGraph
from pmb_scripts.sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE, split_synset_id


NLTK_RESERVED = {
    "all", "exists", "exist",
    "and", "or", "not",
    "true", "false",
    "iff", "implies",
}


def sanitize_ident(s: str, *, prefix: str = "C") -> str:
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    if not s:
        s = "EMPTY"

    if not re.match(r"^[A-Za-z]", s):
        s = f"{prefix}_{s}"

    if s.lower() in NLTK_RESERVED:
        s = f"{prefix}_{s}"

    if re.fullmatch(r"[A-Za-z]", s):
        s = f"{prefix}_{s}"

    return s


def pred_from_synset(token: str) -> str:
    return sanitize_ident(token.replace(".", "_").replace("-", "_"), prefix="p")


def pred_from_role(label: str) -> str:
    return sanitize_ident(label.replace("-", "_"), prefix="p")


def const_from_token(token: str) -> str:
    return sanitize_ident(token, prefix="C")


def var_for_synset(node_id, node_data) -> str:
    token = node_data["token"]
    parsed = split_synset_id(token)
    idx = node_id[1] + 1

    if not parsed:
        return f"x{idx}"

    lemma, pos, _sense = parsed

    if pos == "n" and lemma == "time":
        return f"t{idx}"

    if pos in {"n", "x"}:
        return f"x{idx}"

    return f"e{idx}"


def is_synset(G, node_id) -> bool:
    return G.nodes[node_id]["type"] == SBN_NODE_TYPE.SYNSET


def is_constant(G, node_id) -> bool:
    return G.nodes[node_id]["type"] == SBN_NODE_TYPE.CONSTANT


def is_box(G, node_id) -> bool:
    return G.nodes[node_id]["type"] == SBN_NODE_TYPE.BOX


def get_or_make_term(G, node_id, node_terms: Dict) -> Optional[str]:
    if node_id in node_terms:
        return node_terms[node_id]

    node_data = G.nodes[node_id]

    if node_data["type"] == SBN_NODE_TYPE.SYNSET:
        term = var_for_synset(node_id, node_data)
    elif node_data["type"] == SBN_NODE_TYPE.CONSTANT:
        term = const_from_token(node_data["token"])
    else:
        return None

    node_terms[node_id] = term
    return term


def root_box(G):
    roots = [n for n, deg in G.in_degree() if deg == 0 and is_box(G, n)]
    if not roots:
        raise ValueError("No root BOX node found.")
    return sorted(roots, key=lambda n: n[1])[0]


def box_members(G, box_id) -> List:
    return [
        child
        for _, child, data in G.out_edges(box_id, data=True)
        if data["type"] == SBN_EDGE_TYPE.BOX_CONNECT
    ]


def grouped_child_boxes(G, box_id) -> Dict[str, List]:
    groups: Dict[str, List] = defaultdict(list)

    for _, child, data in G.out_edges(box_id, data=True):
        if data["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            groups[data["token"]].append(child)

    return groups


def unique_preserving_order(items: List[str]) -> List[str]:
    out = []
    seen = set()

    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


def emit_conditionals(G, groups, node_terms: Dict) -> Tuple[List[str], Set]:
    out: List[str] = []
    used: Set = set()

    antecedents = []
    antecedents.extend(groups.get("CONDITION", []))
    antecedents.extend(groups.get("PRECONDITION", []))

    consequents = groups.get("CONSEQUENCE", [])

    n = min(len(antecedents), len(consequents))

    for i in range(n):
        ant_drs = drs_for_box(G, antecedents[i], node_terms)
        cons_drs = drs_for_box(G, consequents[i], node_terms)

        out.append(f"({ant_drs} -> {cons_drs})")
        used.add(antecedents[i])
        used.add(consequents[i])

    return out, used


def emit_alternation(G, groups, node_terms: Dict) -> Tuple[List[str], Set]:
    out: List[str] = []
    used: Set = set()

    alts = groups.get("ALTERNATION", [])

    if len(alts) >= 2:
        drs_items = [drs_for_box(G, alt, node_terms) for alt in alts]

        expr = drs_items[0]
        for item in drs_items[1:]:
            expr = f"({expr} | {item})"

        out.append(expr)
        used.update(alts)

    return out, used


def emit_remaining_child_boxes(
    G,
    groups,
    used_child_boxes: Set,
    node_terms: Dict,
) -> List[str]:
    out: List[str] = []

    for label, boxes in groups.items():
        for child in boxes:
            if child in used_child_boxes:
                continue

            child_drs = drs_for_box(G, child, node_terms)

            if label == "NEGATION":
                out.append(f"-({child_drs})")

            elif label in {"POSSIBILITY", "NECESSITY"}:
                # NLTK DRT has no native modal FOL operator.
                # This preserves content but drops modality.
                out.append(child_drs)

            else:
                # CONTINUATION, ATTRIBUTION, EXPLANATION, CONTRAST,
                # ELABORATION, RESULT, SOURCE, unpaired CONDITION, etc.
                out.append(child_drs)

    return out


def drs_for_box(G, box_id, node_terms: Dict) -> str:
    refs: List[str] = []
    conds: List[str] = []
    local_emitted_synsets: Set = set()

    members = box_members(G, box_id)
    member_set = set(members)

    # 1. Local synset predicates.
    for node_id in members:
        if not is_synset(G, node_id):
            continue

        term = get_or_make_term(G, node_id, node_terms)
        if term is None:
            continue

        refs.append(term)

        if node_id not in local_emitted_synsets:
            pred = pred_from_synset(G.nodes[node_id]["token"])
            conds.append(f"{pred}({term})")
            local_emitted_synsets.add(node_id)

    # 2. Local role/operator edges from source synsets in this box.
    for source, target, data in G.edges(data=True):
        if source not in member_set:
            continue

        edge_type = data["type"]

        if edge_type not in {SBN_EDGE_TYPE.ROLE, SBN_EDGE_TYPE.DRS_OPERATOR}:
            continue

        target_data = G.nodes[target]

        # Skip scope-marker pseudo-edges such as Proposition <1.
        if (
            target_data.get("type") == SBN_NODE_TYPE.CONSTANT
            and str(target_data.get("token", "")).startswith(("<", ">"))
        ):
            continue

        source_term = get_or_make_term(G, source, node_terms)
        target_term = get_or_make_term(G, target, node_terms)

        if source_term is None or target_term is None:
            continue

        pred = pred_from_role(data["token"])
        conds.append(f"{pred}({source_term},{target_term})")

    # 3. Scoped child boxes.
    groups = grouped_child_boxes(G, box_id)
    used_child_boxes: Set = set()

    conditional_conds, conditional_used = emit_conditionals(G, groups, node_terms)
    conds.extend(conditional_conds)
    used_child_boxes.update(conditional_used)

    alt_conds, alt_used = emit_alternation(G, groups, node_terms)
    conds.extend(alt_conds)
    used_child_boxes.update(alt_used)

    conds.extend(
        emit_remaining_child_boxes(
            G,
            groups,
            used_child_boxes,
            node_terms,
        )
    )

    refs = unique_preserving_order(refs)

    return f"DRS([{','.join(refs)}],[{', '.join(conds)}])"


def graph_to_drs(G: SBNGraph) -> str:
    node_terms: Dict = {}
    return drs_for_box(G, root_box(G), node_terms)


def sbn_to_drs(sbn: str, *, is_single_line: bool = False) -> str:
    G = SBNGraph().from_string(sbn, is_single_line=is_single_line)
    return graph_to_drs(G)


def sbn_to_fol(sbn: str, *, is_single_line: bool = False):
    drs = sbn_to_drs(sbn, is_single_line=is_single_line)
    return DrtExpression.fromstring(drs).fol()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/ai-developer/development/drs-to-fol/gold.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--single-line", default=False, action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
        text = json.loads(text)

    if args.single_line:
        items = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        items = [x['sbn'] for x in text]

    data = []

    for i, sbn in enumerate(items):

        drs = sbn_to_drs(sbn, is_single_line=args.single_line)
        fol = DrtExpression.fromstring(drs).fol().__str__()

        data.append({'drs': drs, 'fol': fol, 'sbn': sbn, 'raw': text[i]['raw']})

    if args.output is not None:
        with open(args.output, "w") as file:
            json.dump(data, file, indent=4)
    else:
        print(json.dumps(data))


if __name__ == "__main__":
    main()