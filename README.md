# FINM-33150-Final-Project
FINM 33150 Final Project by George Lord, Max Zhalilo, and Chris Mulligan

SPY data: https://drive.google.com/drive/folders/1C1a70w9k-z78yTSwz2QPKVb-ZJMKi84U?usp=sharing

### Sim Run Instructions

```bash
python src/main.py \
  --spx "Data/spx_clean.parquet" \
  --vix "Data/vix_clean.parquet" \
  --spy "Data/spy_clean.parquet" \
  --kalshi-clean "Data/kalshi_kxinx_clean.parquet" \
  --kalshi-clean "Data/kalshi_kxinxu_clean.parquet" \
  --output "simulation_output.parquet" \
  --out-of-market-spread-ticks 10
```

### TO-DO:
- NEED SOMEONE TO CHECK THIS: Proper short-selling margin rules + Don't allow buying when cash < $1,000 (ie. 10%)
- Explore Leverage for SPY buying
- Adverse Selection Risk Management
    - For example, widening out as a contract approaches expry.