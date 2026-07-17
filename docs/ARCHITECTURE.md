# Architecture and repository boundaries

## Purpose

Mokumitsu owns the domain model and research assumptions for staged renewal of
dense Japanese wooden-residential neighbourhoods. It does not own a general CFD
engine and it does not attempt to be a universal environmental optimizer.

The repository has three layers:

    research core
        district schema and generation
        access, fire, wind and renewal metrics
        joint-project geometry, Pareto and feasibility logic

    optional adapters
        TorchScript FNO checkpoint
        Houdini HIP builders and cached visualization

    verification
        XLB forward checks through houdini-xlb

## Dependency direction

The allowed direction is:

    mokumitsu core -> numpy + shapely + torch
    Houdini builder -> mokumitsu core + hou
    XLB verification script -> mokumitsu core + houdini-xlb

The reverse dependencies are forbidden:

- The package under src/mokumitsu must never import hou.
- houdini-xlb must never import mokumitsu.
- Domain objectives, rights assumptions and renewal rules must not move into
  houdini-xlb.
- A trained checkpoint is an external artifact, not source code.

Torch is currently a package dependency because wind screening and
differentiable FNO evaluation are first-class research features. A later release
may split the wind adapter into an optional extra if the import surface can remain
stable and the additional packaging complexity is justified.

## Data flow

    configuration + seed
        -> MokumitsuDistrict
        -> access and age priorities
        -> fire and FNO screening
        -> individual or joint-renewal candidates
        -> Pareto shortlist
        -> phased feasibility
        -> Houdini visualization
        -> selected candidates verified with XLB

All design geometry is represented in metres and serialized as ordinary Python
dataclasses and JSON-compatible dictionaries. Houdini receives this geometry at
the adapter boundary; it is not the source of truth for the research model.

## Physics boundary

The graph fire model and FNO are screening models. They rank and compare
alternatives cheaply.

XLB is the higher-fidelity wind verification path. It is invoked only for
shortlisted layouts because it is substantially more expensive. A candidate
should not be described as validated merely because it improves the FNO
objective.

The current FNO output is pedestrian-level scalar speed. It does not expose a
full velocity vector or indoor ventilation rate. The Houdini heatmap therefore
shows weighted U/U0, not streamlines and not room air changes.

## Reproducibility boundary

Source-controlled:

- package, CLI, experiment and verification scripts;
- tests and explicit default policies;
- Japanese research notes and related-work review;
- two small HIP samples and their precomputed bgeo.sc playback caches.

External or generated:

- TorchScript checkpoints and training datasets;
- XLB fields and large research outputs;
- regenerated JSON, figures and local Houdini artifacts.

Research output should record the requested and resolved model names, grid size,
checkpoint size and SHA-256 when results are intended for comparison.

## Why Houdini remains in this repository

The Houdini files are thin domain-specific adapters and examples. Keeping them
beside Mokumitsu makes the research sequence inspectable and prevents a second
repository from having to know domain-specific project phases, rights or cost
attributes.

The generic solver connection remains in houdini-xlb. This preserves a clean
boundary:

    Mokumitsu decides what to evaluate.
    Houdini edits and displays geometry.
    houdini-xlb runs and caches XLB.

## Future extension point

The next architectural step is a small evaluator contract with:

- named objectives and constraints;
- geometry and scenario inputs;
- provenance and uncertainty metadata;
- fast preview and expensive verification modes;
- deterministic cache keys.

This interface should first be proven by Mokumitsu's FNO, fire and XLB paths.
Only after a second independent design domain uses the same contract should the
interface and optimizer be extracted into a generic third repository.

See [Project status and roadmap](ROADMAP.md) for the decision criteria.
