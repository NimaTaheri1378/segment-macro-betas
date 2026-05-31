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
- Deep Sets validates the segment-set modeling path, but does not currently
  dominate the tabular benchmark.
- The Set Transformer path has passed a bounded GPU runtime smoke, not a full
  scale empirical comparison.

Forbidden current wording:

- Segment macro betas cause returns.
- Segment disclosures prove investor underreaction.
- The strategy is tradable, capacity-safe, arbitrage-like, or fully orthogonal
  to standard factors.
- Macro-vintage interactions are final before live official macro pulls and
  macro tensor review.
- 2026 holdout performance is known before the holdout protocol is explicitly
  opened.

Every manuscript-facing claim should map to a private manifest, table, or
figure and should be regenerated through `scripts/run_claim_ledger.sh` after
new empirical runs.
