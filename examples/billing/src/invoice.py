"""Invoice totals for the billing service."""


class Invoice(object):
    def __init__(self, customer, items=None):
        self.customer = customer
        if items == None:
            items = []
        self.items = items

    def subtotal(self):
        total = 0
        for i in range(len(self.items)):
            item = self.items[i]
            total = total + item["price"] * item["qty"]
        return total

    def tax(self, rate):
        return round(self.subtotal() * rate, 2)

    def summary(self, rate=0.15):
        return "%s: %d items, total %.2f" % (
            self.customer, len(self.items), self.subtotal() + self.tax(rate))


def load_rates(path):
    """Read 'region,rate' lines into a dict."""
    f = open(path)
    rates = {}
    for line in f.readlines():
        parts = line.strip().split(",")
        if len(parts) == 2:
            rates[parts[0]] = float(parts[1])
    f.close()
    return rates
