from __future__ import annotations

from dataclasses import asdict, dataclass


REGRESSION_TOLERANCE = 0.05
MIN_OVERALL_ACC_DELTA = -0.03


@dataclass(slots=True)
class GateResult:
    gate_name: str
    passed: bool
    value: float
    threshold: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_gates(
    *,
    overall_acc: float,
    lr_mae: float,
    oom_rate: float,
    ece: float,
    overconf: float,
    per_cat: dict[str, float],
    css_integrity: bool,
    current_prod_overall_acc: float,
    current_prod_per_cat: dict[str, float],
) -> list[GateResult]:
    gates: list[GateResult] = []

    gates.append(
        GateResult(
            gate_name="oom_safety",
            passed=oom_rate < 0.01,
            value=float(oom_rate),
            threshold="< 0.01 (hard)",
            detail=f"oom_violation_rate={oom_rate:.4f}",
        )
    )

    delta = float(overall_acc - current_prod_overall_acc)
    gates.append(
        GateResult(
            gate_name="accuracy_no_regression",
            passed=delta >= MIN_OVERALL_ACC_DELTA,
            value=delta,
            threshold=f">= {MIN_OVERALL_ACC_DELTA}",
            detail=f"overall_accuracy_delta={delta:+.4f}",
        )
    )

    gates.append(
        GateResult(
            gate_name="lr_log_mae",
            passed=lr_mae <= 0.35,
            value=float(lr_mae),
            threshold="<= 0.35",
            detail=f"lr_log_mae={lr_mae:.4f}",
        )
    )

    regressions: list[str] = []
    for category, acc in sorted(per_cat.items()):
        prod_acc = float(current_prod_per_cat.get(category, 0.0))
        if float(acc) < (prod_acc - REGRESSION_TOLERANCE):
            regressions.append(f"{category}:{acc:.3f}<{prod_acc:.3f}-0.05")
    gates.append(
        GateResult(
            gate_name="no_category_regression",
            passed=not regressions,
            value=float(len(regressions)),
            threshold="0 regressions",
            detail=f"regressions={regressions or 'none'}",
        )
    )

    gates.append(
        GateResult(
            gate_name="confidence_calibration",
            passed=(ece < 0.10 and overconf < 0.07),
            value=float(ece),
            threshold="ece < 0.10 AND overconf < 0.07",
            detail=f"ece={ece:.4f}, overconf={overconf:.4f}",
        )
    )

    gates.append(
        GateResult(
            gate_name="css_integrity",
            passed=bool(css_integrity),
            value=float(bool(css_integrity)),
            threshold="True (hard)",
            detail="ood_synthetic rows must trigger constrained-safe profile behavior",
        )
    )

    return gates

