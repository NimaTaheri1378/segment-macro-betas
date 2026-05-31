# Methodology

The core panel is a firm-month panel. Each firm-month inherits the latest
publicly usable geographic segment map, converts segment sales to geography
weights, and merges those weights to returns and controls.

The panel activates segment snapshots only after the inferred public date. The
preferred activation clock is Compustat `pdate`/`fdate` matched by `gvkey` and
segment fiscal `datadate`, plus one day. When no filing-date match exists, the
pipeline falls back to the WRDS segment `srcdate` plus one day and records that
source in the manifest.

Primary research layers:

- Exposure-sorted portfolios.
- Monthly cross-sectional regressions.
- Exposure-managed factors.
- LightGBM tabular benchmark.
- Deep Sets and optional Set Transformer extensions for segment sets.

The current baseline layer forms monthly quintile portfolios on
point-in-time foreign-sales exposure, evaluates next-month excess returns,
computes monthly rank ICs, and estimates month-by-month cross-sectional
slopes for the exposure variable.

The factor-robustness layer evaluates model-sorted long-short portfolios
against available benchmark factors, records Newey-West alpha diagnostics, and
adds a simple turnover-cost adjustment. Factor returns are aligned to the
realized next-month return date rather than the portfolio-formation date.

The LightGBM layer uses an expanding time-series validation design. For each
validation year, it trains only on earlier firm-months and predicts
next-month excess returns for that validation year. The primary metric is the
monthly rank IC of predictions; the secondary diagnostic forms monthly
prediction quintiles and tracks the equal-weight Q5 minus Q1 return spread.
The runner defaults to `LGBM_DEVICE_TYPE=auto`, which attempts LightGBM GPU
training and records any CPU fallback in the manifest.

The benchmark reports feature ablations rather than a single headline model.
The core variants compare the full feature set with versions that remove
same-month market factors, remove both market factors and own-return fields,
use segment features only, or use non-segment controls only. This keeps the
diagnostics focused on whether segment disclosure features add signal beyond
standard return, factor, and accounting controls.

The segment-set extension keeps each disclosed geography as a token rather
than collapsing the snapshot immediately to summary statistics. The current
benchmark uses a Deep Sets encoder with geography-token embeddings and
revenue-share weights. It reports `set_only` and `set_plus_controls` variants
under the same expanding yearly validation design as the tabular benchmarks.
The optional `set_transformer` variant replaces Deep Sets pooling with
permutation-equivariant self-attention before pooling. The PyTorch runner
defaults to `SET_DEVICE_TYPE=auto` and uses CUDA when the allocation exposes a
GPU.

The macro-tensor layer joins each segment token to cached macro states by
canonical macro area. It uses `available_date`, `realtime_start`, or
`release_date` when those fields exist, so macro observations are as-of joined
to the firm-month. If a cached macro file lacks vintage or release timing, the
manifest marks the run as not fully vintage-safe instead of silently promoting
it to a final result.

The public macro catalog assigns each official macro series to a macro area and
configured release lag. Global macro states are allowed as a fallback for all
segment-token areas. These configured lags are no-lookahead timing controls;
they are not treated as evidence that the cached series is unrevised historical
vintage data.

The visual pack is generated from private manifests and ignored artifacts. It
collects sample coverage, filing-date activation coverage, exposure
distributions, sector-geography exposure, model-comparison metrics, a firm
explorer snapshot, an HTML dashboard, and a model card with claim guardrails.
