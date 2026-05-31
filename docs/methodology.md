# Methodology

The core panel is a firm-month panel. Each firm-month inherits the latest
publicly usable geographic segment map, converts segment sales to geography
weights, and merges those weights to returns and controls.

The smoke pipeline uses segment source dates as a conservative first activation
check. The full pipeline will replace this with SEC filing acceptance dates
where the WRDS segment tables do not expose a direct public-availability date.

Primary research layers:

- Exposure-sorted portfolios.
- Monthly cross-sectional regressions.
- Exposure-managed factors.
- LightGBM tabular benchmark.
- Deep Sets and Set Transformer extensions for segment sets.

The current baseline layer forms monthly quintile portfolios on
point-in-time foreign-sales exposure, evaluates next-month excess returns,
computes monthly rank ICs, and estimates month-by-month cross-sectional
slopes for the exposure variable.

The LightGBM layer uses an expanding time-series validation design. For each
validation year, it trains only on earlier firm-months and predicts
next-month excess returns for that validation year. The primary metric is the
monthly rank IC of predictions; the secondary diagnostic forms monthly
prediction quintiles and tracks the equal-weight Q5 minus Q1 return spread.

The benchmark reports feature ablations rather than a single headline model.
The core variants compare the full feature set with versions that remove
same-month market factors, remove both market factors and own-return fields,
use segment features only, or use non-segment controls only. This keeps the
diagnostics focused on whether segment disclosure features add signal beyond
standard return, factor, and accounting controls.
