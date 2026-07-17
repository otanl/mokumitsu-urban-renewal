# Changelog

## Unreleased

### Added

- explicit joint-design parameters for position, coverage, aspect, rotation and
  floors, with invalid inputs rejected instead of silently repaired;
- a reusable baseline/design evaluator shared by Python and Houdini;
- deterministic cache keys covering district, geometry, policy, scenarios and
  model identity;
- a live Houdini joint-redevelopment scene with automatic wind, fire, capacity
  and open-space feedback;
- a persistent external NeuralOperator worker with CUDA support, warm baseline
  reuse and content-addressed bgeo.sc display caches;
- headless HIP verification and a worker benchmark script;
- release-hosted portable and accelerated residential FNO checkpoints plus the
  500-sample XLB training dataset;
- an MIT-licensed asset manifest with exact byte sizes and SHA-256 values;
- verified model download, FNO retraining and TorchScript export scripts.

### Changed

- district wind evaluation can reuse static masks and return the visualization
  field in the same pass;
- fire graph construction uses precomputed footprints and Shapely STRtree
  neighbourhood queries.

## 0.1.0 - 2026-07-17

Initial public research snapshot.

### Added

- deterministic organic Mokumitsu district generation and access proxies;
- building-age, structure and fire-resistance cohorts;
- stochastic graph fire-spread screening;
- cardinal wind-rose screening with a TorchScript FNO adapter;
- individual and two-to-four-parcel joint-renewal candidates;
- Pareto comparison across fire, wind, footprint, access, rights and connected
  open space;
- phased rights, relocation, dwelling-capacity and scenario-cost feasibility;
- optional XLB verification through houdini-xlb;
- two cached Houdini timeline examples, including an FNO U/U0 display toggle;
- English and Japanese project documentation, related work, architecture,
  limitations and roadmap.

### Known limitations

- trained FNO checkpoints and datasets are not yet distributed with a public
  artifact manifest;
- synthetic district and feasibility coefficients are not calibrated to a real
  district;
- fire and FNO outputs are screening metrics;
- XLB validation covers selected cases rather than a full multi-seed matrix.
