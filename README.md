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
  --output "simulation_output.parquet"
```

### TO-DO:
- Implement proper short-selling margin rules.
- Check cash / pnl / performance metrics.
- Adverse Selection Risk Management
    - For example, widening out as a contract approaches expry.