# nohup python supervised.py --path config/eunet.yaml > sup.out &
nohup python supervised.py --path config/eunet.yaml > sup.out &
wait
nohup python supervised.py --path  "config/eunet_batch*2.yaml" > sup2.out &