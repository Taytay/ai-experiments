"""Row 140: from blind_v2.json (scripts/build_blind_v2.py, written blind by an agent from the product brief only), the files the scoring
scripts expect: blind_v2_others.json (items with the other-users line in the prompt), blind_v2_ynabrule.json and blind_v2_payeehist.json
(YNAB's current rule and every category filed for the payee before, per item).
usage: uv run python scripts/derive_blind_v2.py
"""
import hashlib
import json

from ai_experiments.paths import PROCESSED

d = json.loads((PROCESSED / "blind_v2.json").read_text())
items = [dict(it, prompt=it["prompt_others"], prompt_ctx=it["prompt_others"]) for it in d["items"]]
sha = hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
(PROCESSED / "blind_v2_others.json").write_text(json.dumps(dict(d, name="blind_v2_others", items=items, sha256=sha), ensure_ascii=False))
(PROCESSED / "blind_v2_ynabrule.json").write_text(json.dumps({it["id"]: it["ynab_rule"] for it in d["items"]}, indent=0))
(PROCESSED / "blind_v2_payeehist.json").write_text(json.dumps({it["id"]: it["payee_hist"] for it in d["items"]}, indent=0))
print("wrote blind_v2_others / _ynabrule / _payeehist for", len(items), "items")
