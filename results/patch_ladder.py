from pathlib import Path
import py_compile
p = Path("experiments/universe.py"); t = p.read_text()
old = """        mc("L1_recall", f"Question: {q}\nAnswer:", [" " + o for o in opts], opts.index(s[attr]), attr=attr)
"""
new = """        mc("L1_recall", f"Question: {q}\nAnswer:", [" " + o for o in opts], opts.index(s[attr]), attr=attr)
    # L1b recall in the *trained* answer format ("X is a T-type.") to separate knowledge from format shift
    for s in [rng.choice(seen) for _ in range(n_per_level)]:
        mc("L1_recall_fmt", f"Question: What type is {s['name']}?\nAnswer:",
           [f" {s['name']} is a {t}-type." for t in TYPE_LIST], TYPE_LIST.index(s["type"]))
"""
assert old in t; p.write_text(t.replace(old, new))
p = Path("experiments/exp_universe_ladder.py"); t = p.read_text()
old = 'STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600\nBS, LR, SEED = 16, 2e-4, 0\n'
new = 'STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600\nLR = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-4\nBS, SEED = 16, 0\n'
assert old in t; t = t.replace(old, new)
old = 'OUT = Path(__file__).parent.parent / "results" / f"universe_{tag}.json"\n'
new = 'OUT = Path(__file__).parent.parent / "results" / f"universe_{tag}_lr{LR:g}.json"\nADAPTER = Path(__file__).parent.parent / "outputs" / f"universe_{tag}_lr{LR:g}_lora"\n'
assert old in t; t = t.replace(old, new)
old = 'model = train_lora(model, tok)\n'
new = 'model = train_lora(model, tok)\nmodel.save_pretrained(ADAPTER); print(f"   saved adapter -> {ADAPTER}")\n'
assert old in t; t = t.replace(old, new)
p.write_text(t)
py_compile.compile("experiments/universe.py", doraise=True); py_compile.compile("experiments/exp_universe_ladder.py", doraise=True)
print("ok")
