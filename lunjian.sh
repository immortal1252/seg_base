python -u universeg_pipeline.py --path full/res_stack_se.yaml
python -u universeg_pipeline.py --path full/universeg_base2_full.yaml
python -u supervised.py         --path full/unet.yaml
python -u supervised.py         --path full/skunet.yaml
python -u supervised.py         --path full/aaunet.yaml
python -u supervised.py         --path full/attnunet.yaml
shutdown -h now