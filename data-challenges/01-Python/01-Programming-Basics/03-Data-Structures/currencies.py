# pylint: disable=missing-docstring

RATES = {
    'USDEUR': '0.85',
    'GBPEUR': '1.13',
    'CHFEUR': '0.86',
    'EURGBP': '0.885'
}


def convert(amount, currency):
    """returns the converted amount in the given currency
    amount is a tuple like (100, "EUR")
    currency is a string
    """
    out = amount[0]
    if amount[1] == currency:
        return out
    else:
        try:
            out *= float(RATES[amount[1]+currency])
        except:
            try:
                out *= (1/float(RATES[currency+amount[1]]))
            except:
                return None
        return round(out)
