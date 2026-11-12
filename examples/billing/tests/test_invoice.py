from invoice import Invoice, load_rates

ITEMS = [{"price": 10.0, "qty": 3}, {"price": 2.5, "qty": 4}]


def test_subtotal_and_tax():
    inv = Invoice("ACME", ITEMS)
    assert inv.subtotal() == 40.0
    assert inv.tax(0.15) == 6.0


def test_no_items():
    assert Invoice("ACME").subtotal() == 0
    assert Invoice("ACME").items == []


def test_summary():
    assert Invoice("ACME", ITEMS).summary() == "ACME: 2 items, total 46.00"


def test_load_rates(tmp_path):
    rates = tmp_path / "rates.csv"
    rates.write_text("QC,0.14975\nON,0.13\n\nbad line\n")
    assert load_rates(rates) == {"QC": 0.14975, "ON": 0.13}
