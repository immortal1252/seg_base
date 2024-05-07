# simple
nohup python abstract_unimatch.py --impl fixmatch.FixMatch --path "./config/eunet_fixmatch.yaml">fixmatch.out 2>&1 &

# ema
# nohup python abstract_unimatch.py --impl fixmatch_ema.FixMatchEMA --path "./config/eunet_fixmatch.yaml">fixmatch_ema.out 2>&1 &