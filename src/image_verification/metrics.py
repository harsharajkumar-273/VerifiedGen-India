BAD_LABELS = {"bad", "fail", "review", "reject"}

def evaluate_reports(reports):
    """Decision-level error rates of the verifier against single gold labels.

    false_accept_rate: share of gold-bad images the verifier passed.
    false_reject_rate: share of gold-good images the verifier did not pass.
    """
    labeled = [r for r in reports if r.gold_label]
    bad = [r for r in labeled if r.gold_label in BAD_LABELS]
    good = [r for r in labeled if r.gold_label == "good"]
    return {
        "labeled_count": float(len(labeled)),
        "false_accept_rate": sum(r.decision == "pass" for r in bad) / len(bad) if bad else None,
        "false_reject_rate": sum(r.decision != "pass" for r in good) / len(good) if good else None,
    }
