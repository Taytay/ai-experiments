"""Open-licence gate (owner, 2026-09-26: only models and code under open licences). open_licence(model_id) walks the model card and
every base_model card on the Hub and stops unless each names a licence in OPEN."""
import time


OPEN = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0"}


def open_licence(model_id):
    """Stop unless the model card and every base model's card name an open licence (OPEN)."""
    from huggingface_hub import model_info
    seen, todo = set(), [model_id]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        for attempt in range(5):  # the Hub API resets connections now and then
            try:
                cd = model_info(m).card_data or {}
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        lic = (cd.get("license") or "").lower()
        assert lic in OPEN, f"{m}: licence {lic!r} is not in the open list {sorted(OPEN)}"
        base = cd.get("base_model") or []
        todo += [base] if isinstance(base, str) else list(base)
    print("licences open:", ", ".join(sorted(seen)), flush=True)
