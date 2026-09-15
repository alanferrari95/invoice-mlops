def assert_min_f1(metrics: dict, min_f1: float) -> str:
    """Raise if f1 is missing or below the promotion threshold."""
    if "f1" not in metrics:
        raise RuntimeError("metrics.json missing key 'f1'")
    f1 = float(metrics["f1"])
    if f1 < min_f1:
        raise RuntimeError(f"f1={f1} below min_f1={min_f1}")
    return f"f1={f1}"
