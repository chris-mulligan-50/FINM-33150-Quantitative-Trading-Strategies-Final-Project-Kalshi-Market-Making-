# FINM-33150-Final-Project
FINM 33150 Final Project by George Lord, Max Zhalilo, and Chris Mulligan

SPY data: https://drive.google.com/drive/folders/1C1a70w9k-z78yTSwz2QPKVb-ZJMKi84U?usp=sharing

### Sim Run Instructions

Make sure you have the `requirements.txt` satisfied (`pip`)

```bash
python main.py \
  --spx "Data/cleaned_parquets/spx_clean.parquet" \
  --vix "Data/cleaned_parquets/vix_clean.parquet" \
  --spy "Data/cleaned_parquets/spy_clean.parquet" \
  --kalshi-clean "Data/cleaned_parquets/kalshi_kxinx_clean.parquet" \
  --kalshi-clean "Data/cleaned_parquets/kalshi_kxinxu_clean.parquet" \
  --output "simulation_output.parquet" \
  --out-of-market-spread-ticks 10
```

Other flags:
```bash
--no-fees
--no-hedge
```