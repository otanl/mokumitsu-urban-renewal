# Contributing

Thank you for improving Mokumitsu. Contributions are welcome in the research
core, validation, documentation, examples and optional adapters.

## Development setup

Use Python 3.12:

    uv venv --python 3.12
    uv pip install -e ".[dev,viz]"

Run the checks before submitting a change:

    .venv\Scripts\python.exe -m pytest -q
    .venv\Scripts\python.exe -m ruff check .
    .venv\Scripts\python.exe -m ruff format --check .

The test suite uses small fake wind models and does not require a trained FNO,
Houdini, XLB or a GPU.

## Repository rules

- Keep src/mokumitsu independent of hou.
- Keep Mokumitsu-specific objectives and planning assumptions out of
  houdini-xlb.
- Preserve deterministic output for a fixed seed.
- State coordinate units and field orientation explicitly.
- Treat fire and FNO outputs as screening metrics, not validated predictions.
- Record model provenance for results intended for comparison.
- Do not commit checkpoints, datasets, generated outputs or local Houdini caches.
  The two curated examples/cache sequences are the only cache exception.

## Research changes

A change to a default coefficient, objective or legal-access proxy should include:

- the rationale and source, if available;
- a test that fixes the intended behavior;
- a note in docs/RESEARCH.md when interpretation changes;
- a limitation statement if the change is not empirically calibrated.

Do not describe a Pareto winner as an optimum in an absolute sense. It is
non-dominated only within the generated alternatives, configured objectives and
screening models.

## Houdini and XLB changes

Houdini builders should produce scenes that play from committed File Cache data
without recomputing FNO or optimization. Rebuild scripts must keep local paths
outside committed source where possible.

XLB verification belongs in scripts and should use the public houdini-xlb API.
Tests for the core package must remain CPU-runnable.

## Language

Code-facing documentation and API names are English. Detailed Japanese planning
and regulatory context may remain Japanese. If a change alters project status,
scope or limitations, update both README.md and README.ja.md.
