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
