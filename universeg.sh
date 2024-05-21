python -u universeg_pipeline.py --path full/bigger/res_stack_se.yaml
python -u supervised.py --path full/bigger/aaunet.yaml
python -u supervised.py --path full/bigger/attnunet.yaml
python -u supervised.py --path full/bigger/unet.yaml
shutdown -h now