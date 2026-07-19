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
        portable and accelerated FNO runtimes (no supported v2 checkpoint yet)
        persistent preview worker
        Houdini HIP builders, live editing and cached visualization

    verification
        gated XLB forward checks through houdini-xlb

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
        -> fire and optional validated-wind screening
        -> individual or joint-renewal candidates
        -> Pareto shortlist
        -> phased feasibility
        -> Houdini visualization
        -> selected candidates verified with XLB

When a checkpoint carrying the v2 physical contract is available, the live
joint-design path uses the same core objects:

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

The graph fire model is a screening model. The FNO adapter is also a screening
path, but there is currently no supported residential checkpoint: the v1
dataset and weights are quarantined after a finite catastrophic XLB sample, an
incorrect vertical-scale contract and checkpoint selection on the test split
were found.

The corrected wind contract uses metres, an isotropic lattice, exact direct
rasterization at the requested XLB grid and an interpolated 1.5 m pedestrian
slice. XLB remains the intended higher-fidelity verification path, but the
current KBC configuration fails the grid-independence gate. Consequently neither
the v1 FNO result nor a corrected-grid XLB result may be described as validated.
See [Wind validation status](WIND_VALIDATION_STATUS.md).

A future accepted FNO will output pedestrian-level scalar speed. It will not
expose a full velocity vector or indoor ventilation rate. Houdini heatmaps show
weighted U/U0, not streamlines and not room air changes. The bundled heatmaps
are historical v1 values and are retained only to inspect the UI and cache flow.

## Reproducibility boundary

Source-controlled:

- package, CLI, experiment and verification scripts;
- tests and explicit default policies;
- Japanese research notes and related-work review;
- three small HIP samples and two precomputed bgeo.sc playback caches.

External or generated:

- release-hosted TorchScript/NeuralOperator checkpoints and training datasets
  (the current v1 release is quarantined and audit-only);
- XLB fields, grid-gate reports and large research outputs;
- regenerated JSON, figures and dynamic live-design bgeo.sc caches.

Large model artifacts remain outside Git, while
[`models/manifest.json`](../models/manifest.json) records the release tag,
license, exact byte sizes, SHA-256 values, architecture and training provenance.
The downloader verifies these values before installing an asset and refuses a
quarantined release unless an explicit audit-only override is supplied. Research
output should still record the requested and resolved model names, physical
domain, lattice shape, pedestrian height, backend signature, checkpoint size and
SHA-256 when results are intended for comparison.

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
