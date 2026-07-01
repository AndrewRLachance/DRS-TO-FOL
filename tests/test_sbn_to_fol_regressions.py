from pmb_scripts.sbn2penman import SBNGraph
from pmb_scripts.sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE
from sbn_to_fol import convert_item, sbn_to_drs, sbn_to_fol, sbn_to_semantic_ir


def edge_matches(edge_data, edge_type, token):
    return edge_data["type"] == edge_type and edge_data["token"] == token


def diagnostic_codes(result):
    return {diagnostic["code"] for diagnostic in result["diagnostics"]}


def test_scope_marker_operator_creates_box_edge_without_phantom_nodes():
    sbn = """person.n.01
NEGATION <1
dog.n.01
"""

    graph = SBNGraph().from_string(sbn)
    node_ids = set(graph.nodes)

    assert all(target in node_ids for _, target in graph.edges)

    negation_edges = [
        (source, target, data)
        for source, target, data in graph.edges(data=True)
        if edge_matches(data, SBN_EDGE_TYPE.BOX_BOX_CONNECT, "NEGATION")
    ]

    assert len(negation_edges) == 1

    _source, target, data = negation_edges[0]
    assert graph.nodes[target]["type"] == SBN_NODE_TYPE.BOX
    assert data["scope_marker"] == "<1"


def test_role_scope_marker_is_skipped_during_drs_and_fol_export():
    sbn = """say.v.01 Proposition <1
dog.n.01
"""

    graph = SBNGraph().from_string(sbn)

    proposition_edges = [
        (source, target, data)
        for source, target, data in graph.edges(data=True)
        if edge_matches(data, SBN_EDGE_TYPE.ROLE, "Proposition")
    ]

    assert len(proposition_edges) == 1

    _source, target, data = proposition_edges[0]
    assert graph.nodes[target]["type"] == SBN_NODE_TYPE.CONSTANT
    assert graph.nodes[target]["token"] == "<1"
    assert data["scope_marker"] == "<1"

    drs = sbn_to_drs(sbn)
    fol = str(sbn_to_fol(sbn))

    assert "say_v_01(e1)" in drs
    assert "dog_n_01(x2)" in drs
    assert "Proposition(" not in drs
    assert "Proposition(" not in fol
    assert "C_1" not in fol

    result = convert_item(sbn, single_line=False)
    assert "skipped_scope_marker_edge" in diagnostic_codes(result)
    assert "proposition_scope_modeled" in diagnostic_codes(result)
    assert "unresolved_proposition_scope" in diagnostic_codes(result)
    diagnostic = next(
        item
        for item in result["diagnostics"]
        if item["code"] == "skipped_scope_marker_edge"
    )
    assert diagnostic["label"] == "Proposition"
    assert diagnostic["scope_marker"] == "<1"

    assert result["semantic"]["proposition_frames"] == [
        {
            "type": "proposition_scope",
            "source_id": "synset:0",
            "source_term": "e1",
            "source_token": "say.v.01",
            "role": "Proposition",
            "scope_marker": "<1",
            "lowered_to_fol": False,
            "target_boxes": [],
        }
    ]
    assert result["semantic"]["modal_frames"] == []
    assert result["semantic"]["discourse_frames"] == []
    assert result["semantic"]["ir"]["referents"][0]["kind"] == "event"


def test_multi_digit_role_offsets_resolve_to_synset_targets():
    sbn = """a.x.01
b.x.01
c.x.01
d.x.01
e.x.01
f.x.01
g.x.01
h.x.01
i.x.01
j.x.01
k.x.01 Role -10
"""

    graph = SBNGraph().from_string(sbn)

    role_edges = [
        (source, target, data)
        for source, target, data in graph.edges(data=True)
        if edge_matches(data, SBN_EDGE_TYPE.ROLE, "Role")
    ]

    assert len(role_edges) == 1

    source, target, _data = role_edges[0]
    assert source == (SBN_NODE_TYPE.SYNSET, 10)
    assert target == (SBN_NODE_TYPE.SYNSET, 0)

    drs = sbn_to_drs(sbn)
    fol = str(sbn_to_fol(sbn))

    assert "Role(x11,x1)" in drs
    assert "Role(x11,x1)" in fol


def test_basic_sbn_to_drs_and_fol_with_name_and_role():
    sbn = """person.n.01 Name "Alice"
dog.n.01 Agent -1
"""

    assert sbn_to_drs(sbn) == (
        "DRS([x1,x2],[person_n_01(x1), dog_n_01(x2), "
        "Name(x1,Alice), Agent(x2,x1)])"
    )
    assert str(sbn_to_fol(sbn)) == (
        "exists x1 x2.(person_n_01(x1) & dog_n_01(x2) & "
        "Name(x1,Alice) & Agent(x2,x1))"
    )

    result = convert_item(sbn, single_line=False)
    assert result["error"] is None
    assert result["drs"] == sbn_to_drs(sbn)
    assert result["fol"] == str(sbn_to_fol(sbn))
    assert result["sbn"] == sbn
    assert result["semantic"]["entity_frames"][0]["synset"] == "person.n.01"
    assert result["semantic"]["name_frames"] == [
        {
            "type": "name",
            "id": "name:2",
            "referent_id": "synset:0",
            "term": "x1",
            "name": "Alice",
        }
    ]


def test_negation_scope_marker_exports_parseable_negated_drs_and_fol():
    sbn = """person.n.01 Name "Alice"
NEGATION <1
dog.n.01
"""

    drs = sbn_to_drs(sbn)
    fol = str(sbn_to_fol(sbn))

    assert drs == (
        "DRS([x1],[person_n_01(x1), Name(x1,Alice), "
        "-(DRS([x2],[dog_n_01(x2)]))])"
    )
    assert fol == (
        "exists x1.(person_n_01(x1) & Name(x1,Alice) & "
        "-exists x2.dog_n_01(x2))"
    )


def test_diagnostics_report_malformed_role_offset_fallback():
    sbn = """dog.n.01 Agent +1
"""

    result = convert_item(sbn, single_line=False)
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "Agent(x1,C_1)" in result["fol"]
    assert "possibly_ill_formed_graph" in codes
    assert "malformed_graph_fallback_edge" in codes


def test_diagnostics_report_unknown_role():
    sbn = """dog.n.01 WeirdRole Alice
"""

    result = convert_item(sbn, single_line=False)
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "WeirdRole(x1,Alice)" in result["fol"]
    assert "unknown_role" in codes

    diagnostic = next(
        item for item in result["diagnostics"] if item["code"] == "unknown_role"
    )
    assert diagnostic["label"] == "WeirdRole"


def test_diagnostics_report_unknown_scopal_operator():
    sbn = """SOME_OPERATOR <1
dog.n.01
"""

    result = convert_item(sbn, single_line=False)
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "dog_n_01(x1)" in result["fol"]
    assert "unknown_scopal_operator" in codes
    assert "unsupported_scopal_structure" in codes


def test_diagnostics_report_modal_content_preserved_without_modal_semantics():
    sbn = """POSSIBILITY <1
dog.n.01
"""

    result = convert_item(sbn, single_line=False)
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "dog_n_01(x1)" in result["fol"]
    assert "unsupported_modal_operator" in codes
    assert "modal_scope_modeled" in codes

    assert result["semantic"]["modal_frames"] == [
        {
            "type": "modal_scope",
            "operator": "POSSIBILITY",
            "source_box_id": "box:0",
            "target_box_id": "box:1",
            "scope_marker": "<1",
            "drs": "DRS([x1],[dog_n_01(x1)])",
            "lowered_to_fol": False,
            "fol_handling": "embedded_content_preserved_modality_dropped",
        }
    ]


def test_semantic_model_resolves_proposition_scope_to_matching_box():
    sbn = """say.v.01 Proposition <1
NEGATION <1
dog.n.01
"""

    result = convert_item(sbn, single_line=False)
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "Proposition(" not in result["fol"]
    assert "proposition_scope_modeled" in codes
    assert "unresolved_proposition_scope" not in codes

    frames = result["semantic"]["proposition_frames"]
    assert len(frames) == 1

    frame = frames[0]
    assert frame["source_term"] == "e1"
    assert frame["scope_marker"] == "<1"
    assert frame["lowered_to_fol"] is False
    assert frame["target_boxes"] == [
        {
            "box_id": "box:1",
            "operator": "NEGATION",
            "drs": "DRS([x2],[dog_n_01(x2)])",
        }
    ]


def test_semantic_model_records_structural_discourse_relation():
    sbn = """CONTINUATION <1
dog.n.01
"""

    result = convert_item(sbn, single_line=False)
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "dog_n_01(x1)" in result["fol"]
    assert "discourse_relation_modeled" in codes
    assert "structural_discourse_relation" in codes

    assert result["semantic"]["discourse_frames"] == [
        {
            "type": "discourse_relation",
            "relation": "CONTINUATION",
            "source_box_id": "box:0",
            "target_box_id": "box:1",
            "scope_marker": "<1",
            "drs": "DRS([x1],[dog_n_01(x1)])",
            "lowered_to_fol": False,
            "fol_handling": "merged_structurally",
        }
    ]


def test_semantic_model_records_embedded_discourse_relation():
    sbn = """COMMENTARY <1
dog.n.01
"""

    result = convert_item(sbn, single_line=False)
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "dog_n_01(x1)" in result["fol"]
    assert "discourse_relation_modeled" in codes
    assert "embedded_discourse_relation" in codes
    assert "unsupported_scopal_structure" not in codes

    assert result["semantic"]["discourse_frames"] == [
        {
            "type": "discourse_relation",
            "relation": "COMMENTARY",
            "source_box_id": "box:0",
            "target_box_id": "box:1",
            "scope_marker": "<1",
            "drs": "DRS([x1],[dog_n_01(x1)])",
            "lowered_to_fol": False,
            "fol_handling": "embedded_drs_content",
        }
    ]


def test_semantic_ir_extracts_temporal_and_equality_frames():
    sbn = """be.v.03 Time +1
time.n.08 EQU now
"""

    result = convert_item(sbn, single_line=False)
    semantic = result["semantic"]

    assert result["error"] is None
    assert any(frame["relation"] == "Time" for frame in semantic["time_frames"])
    assert any(
        frame["relation"] == "EQU" and frame["value"] == "now"
        for frame in semantic["time_frames"]
    )
    assert semantic["equality_frames"] == [
        {
            "type": "equality",
            "id": "equality:3",
            "operator": "EQU",
            "left_id": "synset:1",
            "left_term": "t2",
            "right_id": "constant:0",
            "right_term": "now",
            "value": "now",
        }
    ]


def test_semantic_ir_extracts_non_temporal_equality_without_time_frame():
    sbn = """entity.n.01 EQU ?
"""

    result = convert_item(sbn, single_line=False)
    semantic = result["semantic"]

    assert result["error"] is None
    assert semantic["equality_frames"] == [
        {
            "type": "equality",
            "id": "equality:1",
            "operator": "EQU",
            "left_id": "synset:0",
            "left_term": "x1",
            "right_id": "constant:0",
            "right_term": "EMPTY",
            "value": "?",
        }
    ]
    assert semantic["time_frames"] == []


def test_semantic_ir_extracts_quantity_and_generalized_quantifier_uncertainty():
    sbn = """person.n.01 Quantity ? Unit years
"""

    result = convert_item(sbn, single_line=False)
    semantic = result["semantic"]
    codes = diagnostic_codes(result)

    assert result["error"] is None
    assert "generalized_quantifier_uncertain" in codes
    assert semantic["quantity_frames"] == [
        {
            "type": "quantity",
            "id": "quantity:1",
            "source_id": "synset:0",
            "source_term": "x1",
            "role": "Quantity",
            "value": "?",
        },
        {
            "type": "quantity",
            "id": "quantity:2",
            "source_id": "synset:0",
            "source_term": "x1",
            "role": "Unit",
            "value": "years",
            "unit": "years",
        },
    ]
    assert semantic["generalized_quantifier_frames"] == [
        {
            "type": "generalized_quantifier",
            "id": "generalized-quantifier:1",
            "source_id": "synset:0",
            "quantifier": "?",
            "status": "uncertain",
            "details": {"role": "Quantity"},
        }
    ]


def test_semantic_ir_extracts_accessibility_for_negation_and_conditionals():
    negation = convert_item(
        """person.n.01
NEGATION <1
dog.n.01
""",
        single_line=False,
    )
    assert negation["semantic"]["accessibility_frames"] == [
        {
            "type": "accessibility",
            "id": "accessibility:1",
            "parent_box_id": "box:0",
            "child_box_id": "box:1",
            "operator": "NEGATION",
            "scope_marker": "<1",
            "local_referents": ["x2"],
            "inherited_referents": ["x1"],
        }
    ]

    conditional = convert_item(
        """CONDITION <1
dog.n.01
CONSEQUENCE <1
bark.v.01 Agent -1
""",
        single_line=False,
    )
    assert [
        (frame["operator"], frame["parent_box_id"], frame["child_box_id"])
        for frame in conditional["semantic"]["accessibility_frames"]
    ] == [
        ("CONDITION", "box:0", "box:1"),
        ("CONSEQUENCE", "box:1", "box:2"),
    ]


def test_sbn_to_semantic_ir_returns_document_object():
    document = sbn_to_semantic_ir(
        """person.n.01 Name "Alice"
"""
    )

    assert document.referents[0].kind == "entity"
    assert document.to_dict()["entity_frames"][0]["term"] == "x1"
