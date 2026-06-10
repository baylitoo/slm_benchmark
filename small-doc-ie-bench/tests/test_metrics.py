from docie_bench.benchmark.metrics import score_prediction


def test_score_prediction_nested_value():
    pred = {"invoice_number": {"value": "INV-1"}, "total_ttc": {"amount": "1200.00", "currency": "EUR"}}
    gt = {"invoice_number": "INV-1", "total_ttc.amount": "1200"}
    score = score_prediction(gt, pred)
    assert score["field_correct"] == 2
