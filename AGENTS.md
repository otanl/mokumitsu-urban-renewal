# AGENTS.md

## Project

Mokumitsu is a synthetic research prototype for staged renewal of dense Japanese
wooden-residential neighbourhoods. It combines access, fire, wind, joint
redevelopment, Pareto screening and phased feasibility.

## Boundaries

- src/mokumitsu is the research core and must not import hou.
- houdini contains optional scene builders only.
- XLB verification uses the public houdini-xlb package.
- houdini-xlb must not depend on Mokumitsu.
- FNO and fire results are screening metrics. Do not describe them as validated
  predictions or absolute optima.
- Checkpoints, datasets and generated outputs stay outside Git.
- The two examples and examples/cache sequences are curated exceptions.

## Environment

Use the project virtual environment and Python 3.12:

    .venv\Scripts\python.exe -m pytest -q
    .venv\Scripts\python.exe -m ruff check .
    .venv\Scripts\python.exe -m ruff format --check .

Tests must remain CPU-runnable without Houdini, XLB, a GPU or a trained
checkpoint.

## Invariants

- A fixed seed produces the same district and renewal sequence.
- Geometry uses metres and field arrays use row=y, column=x.
- Wind directions are directions toward which air moves.
- Model family fallback is never implicit.
- Research outputs preserve model and policy provenance.
- Changes to coefficients or proxies require tests and documentation of their
  interpretation and calibration status.

## Documentation

README.md is the English OSS entry point. README.ja.md and docs/RESEARCH.md are
the detailed Japanese research references. Scope or limitation changes must be
reflected in both README files and docs/ROADMAP.md.
