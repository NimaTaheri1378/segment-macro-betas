# Claim Guardrails

This project treats private empirical outputs as diagnostics until the relevant
gates have passed. Public wording should stay evidence-bound.

Allowed current wording:

- The filing-date panel has zero activation-rule violations in the private
  manifest.
- Segment-only LightGBM features show positive cross-sectional ranking signal
  in the filing-date development panel.
- Control-rich LightGBM variants currently produce stronger long-short spread
  diagnostics than the segment-only variant.
- Some LightGBM variants retain positive factor-alpha diagnostics in the cached
  robustness run.
- Official non-FRED macro data from BLS, BEA, and EIA have been pulled into a
  no-lookahead macro tensor for the 2006-2025 development sample.
- The macro-aware `all_plus_macro` LightGBM diagnostic has the strongest
  current long-short spread and factor-alpha diagnostics in the development
  sample.
- The `macro_only` diagnostic is not cross-sectional because the current
  official macro inputs are global monthly states.
- Deep Sets validates the segment-set modeling path, but does not currently
  dominate the tabular benchmark.
- The full Set Transformer path runs on CUDA, but remains a weak diagnostic
  relative to the current LightGBM spread and alpha results.
- The 2026 holdout protocol can be described as frozen only when the private
  holdout manifest reports `status=frozen` and `holdout_opened=false`.

Forbidden current wording:

- Segment macro betas cause returns.
- Segment disclosures prove investor underreaction.
- The strategy is tradable, capacity-safe, arbitrage-like, or fully orthogonal
  to standard factors.
- The full FRED-inclusive macro catalog is complete while the FRED API remains
  rate-limited.
- Macro-vintage or revision-safe interactions are final before true vintage or
  realtime macro sources are reviewed.
- 2026 holdout performance is known before the holdout protocol is explicitly
  opened.
- A frozen holdout protocol is the same as a passed 2026 performance result.

Every manuscript-facing claim should map to a private manifest, table, or
figure and should be regenerated through `scripts/run_claim_ledger.sh` after
new empirical runs.
