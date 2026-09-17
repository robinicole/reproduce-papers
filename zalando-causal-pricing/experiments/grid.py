"""The experiment loop both runners share: cells x evaluate -> one CSV with provenance columns.

A cell is a dict naming a unit of work (model, seed, period...). `evaluate(cell)` returns row dicts.
Every row gets the cell fields, the run's provenance (setting, budgets) and the git hash, so the
report groups by columns and never has to decode a filename. Cells already present in the CSV
(same cell fields and provenance) are skipped, which makes any run resumable and lets the pipeline
add budgets to an existing results file.
"""
import logging
import subprocess
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_grid(out, cells, evaluate, provenance):
    out = Path(out)
    keys = list(cells[0]) + list(provenance)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {tuple(str(r.get(k)) for k in keys) for r in rows}
    sha = git_sha()
    for cell in cells:
        if tuple(str(v) for v in list(cell.values()) + list(provenance.values())) in done:
            continue
        t0 = time.time()
        new = evaluate(cell)
        for r in new:
            r.update(cell); r.update(provenance); r["git_sha"] = sha; r["secs"] = round(time.time() - t0, 1)
        rows += new
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        log.info("done %s in %.0fs", cell, time.time() - t0)
    return pd.DataFrame(rows)
