# Data Contract

The frozen WRDS schema map lives in `configs/schema_map.yml`.

Selected sources:

- Segment geography: `comp_segments_hist_daily.wrds_seg_geo`
- Segment values/sales: `comp_segments_hist_daily.wrds_segmerged`
- CRSP monthly: `crsp.msf_v2`
- CRSP daily: `crsp.dsf_v2`
- CCM link history: `crsp_a_ccm.ccmxpf_linktable`
- Compustat annual fundamentals: `comp.funda`
- Optional WRDS factors: `ff.factors_monthly`

Private extraction output is written as Parquet shards under `data/raw/<run_id>/`.
Every extraction run writes compact logs, a JSON manifest, and a Markdown report
under `runs/<run_id>/`.

The filing-date supplement writes `compustat_filing_dates.parquet` shards next
to the raw annual shards. It contains only `gvkey`, `datadate`, fiscal-year
fields, Compustat `pdate`/`fdate`, and the derived `filing_date`. Panel
construction uses this date plus one day as the preferred segment activation
timestamp, falling back to `srcdate` plus one day when no filing-date match is
available.

The macro tensor builder expects a cached macro Parquet file under
`data/raw/<macro_run_id>/`. The default dataset name is
`macro_fred_monthly.parquet`, with columns `series_id`, `date`, and `value`.
Optional timing columns `available_date`, `realtime_start`, or `release_date`
are used for as-of joins; if none are present, the manifest records an
observation-date fallback and `vintage_safe=false`.
