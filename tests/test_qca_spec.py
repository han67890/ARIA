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
    classification = payload["classification"]
    assert classification["multi_hop_min_entities"] == 2
    assert classification["hop_precedes_aspect"] is True
