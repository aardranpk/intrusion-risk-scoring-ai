import pytest

from src.models.risk_scoring import (
    RiskOutput,
    make_risk_output,
    probability_to_risk,
    risk_to_severity,
)


@pytest.mark.parametrize(
    "prob, expected",
    [(0.0, 0), (0.5, 50), (0.874, 87), (1.0, 100)],
)
def test_probability_to_risk_scales_to_0_100(prob, expected):
    assert probability_to_risk(prob) == expected


@pytest.mark.parametrize("prob, expected", [(-0.3, 0), (1.7, 100)])
def test_probability_to_risk_clamps_out_of_range_values(prob, expected):
    assert probability_to_risk(prob) == expected


@pytest.mark.parametrize(
    "score, expected",
    [
        (0, "Low"),
        (39, "Low"),
        (40, "Medium"),   # boundary
        (69, "Medium"),
        (70, "High"),     # boundary
        (89, "High"),
        (90, "Critical"), # boundary
        (100, "Critical"),
    ],
)
def test_risk_to_severity_bands(score, expected):
    assert risk_to_severity(score) == expected


def test_make_risk_output_combines_score_and_severity():
    out = make_risk_output(0.93)
    assert out == RiskOutput(risk_score=93, severity="Critical", model_probability=0.93)


def test_risk_output_is_immutable():
    out = make_risk_output(0.2)
    with pytest.raises(Exception):
        out.risk_score = 99
