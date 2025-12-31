from dataclasses import dataclass

@dataclass(frozen=True)
class RiskOutput:
    risk_score: int          # 0..100
    severity: str            # Low/Medium/High/Critical
    model_probability: float # 0..1

def probability_to_risk(prob: float) -> int:
    """
    Convert model predicted probability (0..1) to a 0..100 risk score.
    Kept simple & explainable for a 2-3 day build.
    """
    prob = max(0.0, min(1.0, float(prob)))
    return int(round(prob * 100))

def risk_to_severity(risk_score: int) -> str:
    """
    Severity bands tuned for SOC-style triage.
    Adjust later based on false positive tolerance.
    """
    if risk_score >= 90:
        return "Critical"
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"

def make_risk_output(prob: float) -> RiskOutput:
    score = probability_to_risk(prob)
    return RiskOutput(
        risk_score=score,
        severity=risk_to_severity(score),
        model_probability=float(prob),
    )
