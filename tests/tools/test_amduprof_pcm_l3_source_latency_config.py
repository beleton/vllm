import json
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "test_results" / "amduprof_pcm_l3_source_latency.conf"
OFFICIAL_L3_METRICS_PATH = Path(
    "/opt/AMDuProf_5.2-606/bin/AMDPerf/data/0x1a_0x1/l3_metrics.json")
OFFICIAL_CONFIG_PATH = Path(
    "/opt/AMDuProf_5.2-606/bin/Data/Config/0x1a_0x1.conf")


def _metric_expression_by_name(config_path: Path) -> dict[str, str]:
    root = ET.fromstring(config_path.read_text(encoding="utf-8"))
    expressions: dict[str, str] = {}
    for metric in root.findall(".//metric"):
        name = metric.attrib.get("name")
        expression = metric.attrib.get("expression")
        if name and expression:
            expressions[name] = expression
    return expressions


def _event_ctl_by_name(config_path: Path) -> dict[str, str]:
    root = ET.fromstring(config_path.read_text(encoding="utf-8"))
    ctls: dict[str, str] = {}
    for event in root.findall(".//event"):
        name = event.attrib.get("name")
        ctl = event.attrib.get("ctl")
        if name and ctl:
            ctls[name] = ctl
    return ctls


def _official_l3_metric_expression_by_name(metrics_path: Path) -> dict[str, str]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    expressions: dict[str, str] = {}
    for metric in data.get("metric", []):
        name = metric.get("Name")
        expression = metric.get("Expression")
        if name and expression:
            expressions[name] = expression
    return expressions


def _normalize_expression(expression: str) -> str:
    return "".join(expression.split())


def test_official_json_uses_retired_instruction_normalization_for_l3_access_pti():
    official_metrics = _official_l3_metric_expression_by_name(
        OFFICIAL_L3_METRICS_PATH)

    assert _normalize_expression(
        official_metrics["L3 Cacheable accesses PTI T0"]) == (
            _normalize_expression(
                "((l3_lookup_state.l3_lookup_mask."
                "all_coherent_accesses_to_l3_l3_oa_accessed_l3)"
                "*(1000))/(OsUserInst)"))


def test_custom_l3_pti_formulas_match_official_builtin_xml_config():
    config_metrics = _metric_expression_by_name(CONFIG_PATH)
    official_config_metrics = _metric_expression_by_name(OFFICIAL_CONFIG_PATH)

    assert _normalize_expression(config_metrics["L3 Access (pti)"]) == (
        _normalize_expression(official_config_metrics["L3 Access (pti)"]))
    assert _normalize_expression(config_metrics["L3 Miss (pti)"]) == (
        _normalize_expression(official_config_metrics["L3 Miss (pti)"]))


def test_custom_l3_pti_formulas_use_official_l3_retired_instruction_alias():
    config_metrics = _metric_expression_by_name(CONFIG_PATH)
    access_expression = _normalize_expression(config_metrics["L3 Access (pti)"])
    miss_expression = _normalize_expression(config_metrics["L3 Miss (pti)"])

    assert access_expression == _normalize_expression(
        "$L3Access * 1000 / $RetdInstL3")
    assert miss_expression == _normalize_expression(
        "$L3Miss * 1000 / $RetdInstL3")
    assert "$RetdInst" not in access_expression.replace("$RetdInstL3", "")
    assert "$RetdInst" not in miss_expression.replace("$RetdInstL3", "")


def test_custom_config_collects_raw_retired_instructions_for_l3_pti():
    config_events = _event_ctl_by_name(CONFIG_PATH)

    assert config_events["$RetdInst2"] == "0x4300C0"
