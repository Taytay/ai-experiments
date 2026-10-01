"""Writes scripts/modal_jobs/r98.json, r99.json and r101.json (PLAN rows 98, 99, 101; 2026-09-28). Run from the repo root."""
import json
import os
import subprocess

UV = 'uv run --frozen --with transformers==5.17.0 --with flash-linear-attention --with "peft>=0.21" --with torch==2.13.0 --with torchvision==0.28.0'
ALL = ",".join(str(u) for u in range(200))
RECIPE = dict(DBEP="0.5", FOLD="0", RENAME="0.5", MICRO="16", STEPS="800", ALL_LABELS="1", AUX_LM="1", LABELS="rand26", MISLEAD="mislead_v1",
              ALT="0.1", ALT_SOFT="1", LOOKUP="0.1", OVERRIDE="0.1", EVFREE="0.1", EVFREE_MODE="soft")
FINAL = "decider_decider-4b_none_h100bf16st800{s}_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand26_ev10soft_lora"
POLICIES = ["recent", "self", "self_kind", "self_cover", "kind_cover", "all_kind", "all_embrec", "all_embname", "all_kind_undated", "all_kind_b48"]


def score(model, adapter, items_set, users=None):
    u = f" USERS={users}" if users else ""
    a = f" ADAPTER={adapter}" if adapter else ""
    return f"FAMILY=decider MODEL={model}{a} CONDS=noctx LABELS=rand26 ITEMS_SET={items_set}{u} {UV} python scripts/exp_decision_models.py"


# row 98 stage 1: the final recipe (two seeds) and the untrained decider-4B on every history-slice policy
r98 = [dict(tag=f"r98-final-s{k}", env=dict(ADAPTERS_FROM=f"r89-alt10-s{k}", FAMILY="decider", MODEL="Mapika/decider-4b"),
            cmds=[score("Mapika/decider-4b", FINAL.format(s="" if k == 0 else "s1"), f"real7_v2_{p}", ALL) for p in POLICIES]) for k in (0, 1)]
r98.append(dict(tag="r98-untrained", env=dict(FAMILY="decider", MODEL="Mapika/decider-4b"),
                cmds=[score("Mapika/decider-4b", "", f"real7_v2_{p}", ALL) for p in ("recent", "self", "all_kind")]))

# row 99: encoders trained on the decoder recipe's episodes (DEC_EPISODES=1), scored on REAL-6 fold 0, novel names, REAL-7 v1 and v2 all_kind
ENC_ENV = dict(DEC_EPISODES="1", DBEP="0.5", FOLD="0", RENAME="0.5", MISLEAD="mislead_v1", ALT="0.1", ALT_SOFT="1", LOOKUP="0.1", OVERRIDE="0.1",
               ALL_LABELS="1", RUN_TAG="h100", LLM_BASE="Qwen/Qwen3.5-2B")
ARMS = [("mbert", dict(ARCH="mask", INIT="mbert", STEPS="3000")), ("laya", dict(ARCH="mask", INIT="laya", STEPS="3000")),
        ("gli", dict(ARCH="gliclass", INIT="large", STEPS="3000")), ("ettin1b", dict(ARCH="mask", INIT="ettin1b", STEPS="3000")),
        ("eurobert", dict(ARCH="mask", INIT="eurobert", STEPS="6000", BATCH="8"))]
r99 = []
for tag, extra in ARMS:
    env = dict(ENC_ENV, **extra)
    name = subprocess.run(["uv", "run", "python", "-c", "import sys; sys.path.insert(0,'scripts'); import exp_encoder_mask as E; print(E.NAME)"],
                          env=dict(os.environ, **env), capture_output=True, text=True).stdout.strip().splitlines()[-1]
    sets = ["real6_v1_novel", "real7_v1", "real7_v2_all_kind", "real7_v2_recent"]
    cmds = ["uv run --frozen python scripts/exp_encoder_mask.py"] + [f"STEPS=0 LOAD_FROM={name} ITEMS_SET={s} uv run --frozen python scripts/exp_encoder_mask.py" for s in sets]
    r99.append(dict(tag=f"r99-{tag}", env=env, cmds=cmds))

# row 101: newer small open decoders with the final recipe (one seed), read on every set
SETS = ["", "real6_v1_novel", "mislead_v1", "override_v1", "alternation_v1"]
r101 = []
for tag, model in [("minicpm5", "openbmb/MiniCPM5-2B"), ("granite4m", "ibm-granite/granite-4.0-micro"), ("smollm3", "HuggingFaceTB/SmolLM3-3B"),
                   ("phi4mini", "microsoft/Phi-4-mini-instruct")]:
    ad = f"slot_{model.split('/')[-1]}_none_h100bf16st800_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_aux100_labrand26_ev10soft_lora"
    cmds = [f"{UV} python scripts/exp_decider_finetune.py"] + [score(model, ad, s) for s in SETS] + [score(model, ad, s, ALL) for s in ("real7_v1", "real7_v2_all_kind")]
    r101.append(dict(tag=f"r101-{tag}", env=dict(RECIPE, MODEL=model, RUN_TAG="h100bf16st800", SEED="0"), cmds=cmds))

for n, j in (("r98", r98), ("r99", r99), ("r101", r101)):
    json.dump(j, open(f"scripts/modal_jobs/{n}.json", "w"), indent=1)
    print(n, len(j), "jobs")
print([j["cmds"][1].split("LOAD_FROM=")[1].split()[0] for j in r99])

# row 98 stage 2: the final recipe plus REAL-7-style training users with dated history slices (all_kind, or a mix of policies), two seeds
r98b = []
for pol in ("all_kind", "mix"):
    for k in (0, 1):
        tag = "h100bf16st800" + ("" if k == 0 else "s1")
        ad = f"decider_decider-4b_none_{tag}_f0_ren50_dbep50_mislead_v1_alt10s_lk10_ov10_r7{pol}_aux100_labrand26_ev10soft_lora"
        cmds = [f"{UV} python scripts/exp_decider_finetune.py"] + [score("Mapika/decider-4b", ad, s) for s in SETS] + \
               [score("Mapika/decider-4b", ad, f"real7_v2_{p}", ALL) for p in ("recent", "self", "all_kind", "all_embrec", "all_kind_b48", "all_kind_undated")]
        r98b.append(dict(tag=f"r98b-{pol}-s{k}", env=dict(RECIPE, MODEL="Mapika/decider-4b", RUN_TAG=tag, SEED=str(k), R7TRAIN=pol), cmds=cmds))
json.dump(r98b, open("scripts/modal_jobs/r98b.json", "w"), indent=1)
print("r98b", len(r98b), "jobs", r98b[0]["cmds"][1][:200])


# row 99 again (r99b): fresh episodes on every pass (the first run cycled one draw of 2,250 and the encoders memorised it), RUN_TAG h100fresh
def r99_jobs(run_tag, prefix):
    out = []
    for tag, extra in ARMS:
        env = dict(ENC_ENV, **extra, RUN_TAG=run_tag)
        name = subprocess.run(["uv", "run", "python", "-c", "import sys; sys.path.insert(0,'scripts'); import exp_encoder_mask as E; print(E.NAME)"],
                              env=dict(os.environ, **env), capture_output=True, text=True).stdout.strip().splitlines()[-1]
        sets = ["real6_v1_novel", "real7_v1", "real7_v2_all_kind", "real7_v2_recent"]
        cmds = ["uv run --frozen python scripts/exp_encoder_mask.py"] + [f"STEPS=0 LOAD_FROM={name} ITEMS_SET={s} uv run --frozen python scripts/exp_encoder_mask.py" for s in sets]
        out.append(dict(tag=f"{prefix}-{tag}", env=env, cmds=cmds))
    return out


if __name__ == "__main__" and os.environ.get("ONLY") == "r99b":
    json.dump(r99_jobs("h100fresh", "r99b"), open("scripts/modal_jobs/r99b.json", "w"), indent=1)
