from collections import defaultdict
def evaluate_reports(reports):
    labeled=[r for r in reports if r.gold_label]
    bad={"bad","fail","review"}; accepted=[r for r in labeled if r.decision=="pass"]; rejected=[r for r in labeled if r.decision!="pass"]
    false_accept=sum(r.gold_label in bad for r in accepted)/len(labeled) if labeled else 0.0
    false_reject=sum(r.gold_label=="good" and r.decision!="pass" for r in labeled)/len(labeled) if labeled else 0.0
    return {"labeled_count":float(len(labeled)),"false_accept_rate":false_accept,"false_reject_rate":false_reject}

