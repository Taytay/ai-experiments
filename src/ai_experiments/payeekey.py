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


_PREFIX_V2 = re.compile(r"^(?:(?:SALE|RETURN|REFUND|PURCHASE RETURN|RECURRING|PREAUTHORIZED|PRE-AUTHORIZED)\s+)+")
_WEB = re.compile(r"(?:\.COM|\.NET|\.CO|\.ORG|\.IO)\b.*$")


def payee_key_v2(text):
    """payee_key plus (2026-10-06, owner: "What if they go to Williams fuel #5, pos debit Williams fuel store 111"): a leading Sale / Return /
    Refund word dropped (statement_patterns_v1 measures them), web suffixes (.com ...) cut, hyphens and apostrophes read as nothing
    ("WILLIAMS-SONOMA" = "WILLIAMSSONOMA"), and the key's words joined, so run-together strings ("WILLIAMSFUEL") and spaced ones meet. The
    original payee_key is unchanged (earlier rows use it)."""
    t = _WEB.sub("", text.upper().replace("&", " AND "))
    t = _PREFIX_V2.sub("", _PREFIX.sub("", t))
    k = payee_key(t)
    return k.replace(" ", "")
