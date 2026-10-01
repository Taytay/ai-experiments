"""Row 150: a payee key from a raw statement string, for matching the same payee across users (YNAB payees are per-user entities; the
import string is the shared signal). Drops card-network / processor prefixes, dates, store numbers, phone numbers, URLs' tails and
trailing locations, then keeps the first two words."""
import re

_PREFIX = re.compile(r"^(?:(?:APLPAY|APPLE PAY|POS DEBIT|POS PURCHASE|POS|DEBIT CARD PURCHASE|DEBIT|CHECKCARD \d+|CHECKCARD|PURCHASE AUTHORIZED ON \d\d/\d\d|"
                     r"PURCHASE|RECURRING PAYMENT|ACH CREDIT|ACH DEBIT|ACH|DBT CRD \d+|DBT CRD|CARD PURCHASE|CARD \d+|ONLINE|"
                     r"SQ \*|SQ\*|TST\*|TST \*|PY \*|PY\*|SP \*|SP |CLV\*|IN \*|PP\*|PAYPAL \*)\s*)+")
_GENERIC = {"THE", "LA", "EL", "LE", "MY", "AND", "OF", "DR", "MR", "MRS", "ST", "NEW"}


def payee_key(text):
    t = text.upper().replace("&", " AND ")
    t = _PREFIX.sub("", t)
    t = re.split(r"\s{2,}|#|\*| \d{3,}|\d{2}/\d{2}", t)[0]
    words = [w for w in re.sub(r"[^A-Z ]", "", t).split() if len(w) > 1 or w in ("A",)]
    words = [w for w in words if w not in _GENERIC] or words
    if not words:
        return ""
    return " ".join(words[:2])
