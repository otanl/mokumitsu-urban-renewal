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
        NeuralOperator training checkpoint + persistent preview worker
        Houdini HIP builders, live editing and cached visualization

    verification
        XLB forward checks through houdini-xlb

## Dependency direction

The allowed direction is:

    mokumitsu core -> numpy + shapely + torch
    Houdini builder -> mokumitsu core + hou
    live worker -> mokumitsu core + optional neuraloperator
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

The live joint-design path uses the same core objects:

    Houdini parameters
        -> JSON-lines request to persistent project-Python worker
        -> parameterized_joint_redevelopment
        -> evaluate_joint_design
        -> FNO + graph-fire screening
        -> metrics + downsampled U/U0 returned to Houdini
        -> content-addressed bgeo.sc display cache

The worker exists because Houdini's embedded Python is not the project's CUDA
environment. Its process keeps the model, synthetic district, baseline analysis
and static masks warm. It is an acceleration adapter, not a second research
implementation. Disabling it uses the same evaluator with the portable
TorchScript model in Houdini.

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
- three small HIP samples and two precomputed bgeo.sc playback caches.

External or generated:

- release-hosted TorchScript/NeuralOperator checkpoints and training datasets;
- XLB fields and large research outputs;
- regenerated JSON, figures and dynamic live-design bgeo.sc caches.

Large model artifacts remain outside Git, while
[`models/manifest.json`](../models/manifest.json) records the release tag,
license, exact byte sizes, SHA-256 values, architecture and training provenance.
The downloader verifies these values before installing an asset. Research output
should still record the requested and resolved model names, grid size,
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

## Evaluator extension point

The first concrete evaluator contract is now implemented by the joint-design
baseline/evaluation objects. It provides:

- named wind, fire, floor-area, access and open-space results;
- explicit geometry, scenario and policy inputs;
- model identity and deterministic cache keys;
- the same callable path for tests, scripts and Houdini.

Provenance display, uncertainty metadata and an explicit expensive-verification
result type still need to be added. The interface should next be proven across
Mokumitsu's FNO, fire and XLB paths. Only after a second independent design
domain uses the same contract should the interface and optimizer be extracted
into a generic third repository.

See [Project status and roadmap](ROADMAP.md) for the decision criteria.
