# reproduce-papers: conventions

One folder per reproduction. Each folder is self-contained and follows the same layout:

```
<reproduction>/
  README.md        the write-up: methods, deviations from the papers, results table, assessment
  requirements.txt
  models/          model code: schema.py (data contract), heads.py, nn.py, one file per model, registry.py
  data/            simulators and data preparation; raw downloads live under data/ and are ignored
  experiments/     grid.py (shared loop), one runner per dataset, pipeline.py (every cell of the
                   study), report.py (results/*.csv -> RESULTS.md and the README table)
  scripts/         run_all.sh, a one-liner onto pipeline.py
  tests/           pytest; fast, CPU-only
  papers/          fetch.sh downloads the papers and converts them to markdown; outputs ignored
  results/         generated; ignored
```

Rules:
- Papers, raw data, results and logs are never committed. Only the scripts that produce them are.
- The README keeps the headline results table and its discussion; `report.py` rewrites the table
  between `<!-- RESULTS -->` markers, so re-read the prose after every regeneration.
- Entry points under `experiments/` and `tests/` put `models/`, `data/` and `experiments/` on
  `sys.path`; modules import each other by bare name. Run everything from the reproduction folder.
- Models are registered by name in `models/registry.py` (constructor, display name, neural flag);
  runners, report, pipeline and tests read that and nothing else. Behaviour switches are
  constructor arguments, not strings to parse.
- Results are one CSV per dataset with provenance columns (setting, budgets, git hash). Rows, not
  filenames, say what a number is; `grid.run_grid` skips cells already present.
- Datasets enter through a `Panel` (see `models/schema.py`), which uses Nixtla's variable taxonomy:
  target, treatment, historical exogenous (context only), future exogenous (context and horizon),
  and statics. `Panel.from_long` takes a long `unique_id`/`ds`/`y` frame. Adding a dataset means
  declaring columns, not editing model code; keep it that way, and keep `tests/test_schema.py`
  passing, since it pins the leak-relevant part of the contract.
- Commands, per reproduction: `pytest` (tests), `papers/fetch.sh` (papers), `scripts/run_all.sh`
  (the whole study, resumable, hours on a GPU), `python experiments/report.py` (tables).
- Network construction is seeded as well as training; a run with the same seed reproduces exactly.
- The GPU on this machine is shared with an unrelated long-running Ollama server that holds most of
  its memory. Run one training job at a time; concurrent jobs slow every fit by an order of
  magnitude and cause out-of-memory errors. Tests run on CPU.
- Step-by-step reproduction instructions, gotchas and the expected result pattern for each
  reproduction live in `.claude/skills/reproduce-<name>/`.

## How to work here

Bias toward caution over speed; for trivial edits use judgement.

- **Think before coding.** State assumptions up front. When a request has more than one reading,
  name the readings and pick one aloud, or ask. When a simpler route exists, say so and push back.
- **Simplicity first.** The minimum code that solves the stated problem: no abstraction for
  single-use code, no configurability nobody asked for, no handling of impossible cases. If a
  senior engineer would call it overcomplicated, rewrite it shorter.
- **Surgical changes.** Every changed line traces to the request. Match the existing style, leave
  neighbouring code and comments as they are, and remove only the orphans your own change created.
  Mention pre-existing dead code; leave it in place.
- **Goal-driven execution.** Turn each task into a check that can fail: a test for a bug before
  the fix, `pytest` green before and after a refactor, the expected result pattern in the skill
  for a rerun. For multi-step work, write the steps with their checks first, then loop until each
  check passes.
