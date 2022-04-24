# export WANDB_TAGS=eve_p
# python eva.py --p=1 &
# python eva.py --p=0.75
# python eva.py --p=0.5 &
# python eva.py --p=0.25
# python eva.py --p=0

export WANDB_TAGS=eve_m7
python eva.py --seed=0&
python eva.py --seed=1
python eva.py --seed=2&
python eva.py --seed=3
python eva.py --seed=4&
python eva.py --seed=5
python eva.py --seed=6&
python eva.py --seed=7
python eva.py --seed=8&
python eva.py --seed=9

