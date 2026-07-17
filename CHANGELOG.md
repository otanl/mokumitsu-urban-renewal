# Changelog

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
