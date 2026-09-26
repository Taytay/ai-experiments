# GPU chains

One shell script per PLAN row that needs a sequence of GPU runs. Each script runs from the repo root, executes its steps one after
another through `run <label> <command...>`, and appends `start` / `end` lines (with the exit code and an error count) to
`logs/gpu<row>.log`, with each step's own output in `logs/gpu<row>_<label>.log`. `logs/` is gitignored. Scoring steps that need an
adapter are guarded with `[ -d models/adapters/<name> ]`, so a failed training step does not fall through into training a fresh one.

    nohup scripts/chains/chain_r37.sh > logs/chain_r37.out 2>&1 &
    tail -f logs/gpu37.log

A chain that waits for another one (`chain_r38.sh`) polls the earlier chain's log for its `CHAIN DONE` line. Never run tests, smokes or
profiling on the GPU while a chain runs (it triples step times and once gave a false diagnosis, REPORT.md 31). To stop a chain, kill
the chain script first (`pkill -f "[c]hain_r37"`), then the running python process; finished steps have already written their outputs.
To resume, delete the finished lines from the script (or comment them out) and start it again.
