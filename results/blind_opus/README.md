# Blind strong-reader ceilings

`scripts/blind_ceiling.py write <set> <variants>` samples items and writes the prompt files; each file is given to a fresh
subagent (Opus 5.5, 2026-09-26) with only this instruction, and its reply is saved as `answers_<variant>.txt`:

> You are helping with a small reasoning exercise. Read the file <items file> and nothing else (do not open any other file, do
> not search the filesystem or the web, do not run code). The file contains 60 items, each headed "### item N". Each item shows one
> person's budgeting setup: first a line "Categories: ..." listing the category names this person uses, then some of their past
> bank-card transactions each followed by the category the person filed it under, and finally one new transaction followed by
> "Category:" with no answer. Category names are this person's own; some are ordinary words and some are invented words whose
> meaning you can only work out from the person's past examples. For each item, decide which of that person's categories they would
> most likely file the final transaction under. Your answer must be one of the names in that item's "Categories:" line, spelled
> exactly as there. Reply with exactly 60 lines, one per item, in this format and nothing else before or after:
> item N | <category name> | <a reason of at most 12 words>

`score <set>` prints the accuracies. One run per variant; about 60 items, so read to +-6 to 10 points.
