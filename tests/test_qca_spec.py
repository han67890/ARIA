import json
from pathlib import Path

from openrlhf.models.modeling_aria import _QCA_RULE_WEIGHTS


def test_qca_rule_table_is_complete_and_single_sourced():
    path = Path(__file__).parents[1] / "openrlhf" / "configs" / "qca_rules.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = {
        *(f"H{index:02d}" for index in range(1, 15)),
        *(f"A{index:02d}" for index in range(1, 13)),
        *(f"S{index:02d}" for index in range(1, 13)),
    }
    assert set(_QCA_RULE_WEIGHTS) == expected_ids
    assert _QCA_RULE_WEIGHTS == {
        rule["id"]: float(rule["weight"]) for rule in payload["rules"]
    }
    assert payload["classification"]["conflict_precedence"] == [
        "explicit_hop:H02-H05",
        "explicit_aspect:A01-A06",
        "remaining_hop",
        "remaining_aspect",
        "simple",
    ]
