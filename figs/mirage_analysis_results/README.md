# MIRAGE-APP×ACT-2024 EDA Results

## Summary

This directory contains the complete exploratory data analysis results for the MIRAGE-APP×ACT-2024 dataset.

**Generated Files:**
- **48 PNG visualizations** covering 9 analysis sections
- **5 CSV summary tables** with key statistics

**Current Run:** 1% sample (1,062 flows from 106,157 total)

## Analysis Sections

### Section 1: Temporal Coverage & Session Distribution (6 plots)
- 1.1a: Sessions per month
- 1.1b: Flows per week  
- 1.2: Sessions by app × time heatmap
- 1.3: Daily activity timeline
- 1.4: App distribution over time (stacked area)
- 1.5: App version updates timeline

### Section 2: Protocol Evolution (4 plots)
- 2.1: Protocol distribution by quarter (stacked bar)
- 2.2: Protocol proportions evolution (lines)
- 2.3: Protocol × app × time heatmaps (20 apps)
- 2.5: Protocol evolution per app (top 9)

### Section 3: Traffic Statistical Feature Drift (8 plots)
- 3.1: Packet size distribution ridge plot
- 3.2: Flow duration box plots (quarter/year)
- 3.3: IAT violin plots
- 3.4: Mean packet size time series with 95% CI
- 3.5: Mean flow duration time series with 95% CI
- 3.6: Packet size percentiles heatmap
- 3.7: TCP window size evolution
- 3.8: Flow size vs duration scatter

### Section 4: App Version Impact (6 plots)
- 4.1: App version timeline (top 5 apps)
- 4.2: Version impact box plots (4 features)
- 4.3: Feature statistics by version (normalized heatmap)
- 4.4: Version comparison histograms
- 4.5: Feature scatter matrix by version
- Statistical tests printed to console

### Section 5: Activity-Level Temporal Stability (7 plots)
- 5.1: Activity distribution over time (stacked bar)
- 5.2: Packet size per activity over time (6 activities)
- 5.3: Activity × time heatmap
- 5.4: Packet size by activity ridge plots
- 5.5: Flow characteristics per activity by year
- 5.6: Activity prevalence over time (lines)
- 5.7: Multi-activity sessions

### Section 6: Network Infrastructure Changes (7 plots)
- 6.1: Top 20 ports over time
- 6.2: Port usage early vs late periods
- 6.3: App × port × time heatmap
- 6.4: TCP MSS box plots (upstream/downstream)
- 6.5: TCP Window Scale box plots
- 6.6: Connection establishment patterns
- 6.7: Upstream vs downstream traffic scatter

### Section 7: Covariate Shift Quantification
- Placeholder section for advanced statistical metrics

### Section 8: Device-Specific Patterns (6 plots)
- 8.1: Device usage stacked area chart
- 8.2: Device × app heatmap
- 8.3: Device → app version mapping (table)
- 8.4: Device × app × time heatmap
- 8.5: Feature distributions by device
- 8.7: Device activity Gantt chart

### Section 9: Data Quality Checks (7 plots)
- 9.1: Labeling type over time (exact vs most-common)
- 9.2: Exact labeling rate per month
- 9.3: Session duration distribution
- 9.4: Flows per session over time
- 9.5: Missing data by field × time heatmap
- 9.6: Session duration vs flow count scatter
- 9.7: Background traffic indicators

## Summary Tables (CSV)

1. **year_summary.csv** - Dataset overview by year
2. **app_version_changes.csv** - App version transition log (1031 transitions)
3. **protocol_by_quarter.csv** - Protocol distribution percentages
4. **feature_stability.csv** - Feature stability ranking
5. **activity_by_quarter.csv** - Activity distribution matrix

## Key Findings

- **Temporal span:** 4.2 years (2019-10-03 to 2023-12-14)
- **Apps with version changes:** 15 out of 31 (48.4%)
- **Protocol shift (TVD):** 0.1319 (moderate covariate shift)
- **Packet size change:** -4.49%
- **Flow duration change:** +162.26%
- **Exact labeling rate:** 68.3%

## Running with Full Dataset

To analyze the complete dataset (all 106,157 flows), edit `mirage_eda_complete.py`:

```python
# Comment out or remove these lines (around line 102-104):
# SAMPLE 1% FOR TESTING
# print(f"Original dataset: {len(df):,} flows")
# df = df.sample(frac=0.01, random_state=42)
# print(f"Sampled to: {len(df):,} flows (1%)")
```

Then run:
```bash
source /home/anatbr/students/noamshakedc/env/anaconda3/bin/activate ml2
python mirage_eda_complete.py
```

**Note:** Full dataset analysis will take significantly longer and use more memory.
