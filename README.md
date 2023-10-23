# DreamRec

This is the implementation of our NeurIPS 2023 paper:

> Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, and Xiangnan He. Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion. In NeurIPS 2023.

## Reprocuce

### YooChoose Data

```
python -u DreamRec.py --data yc --cuda 0 --timesteps 500 --lr 0.001 --beta_sche exp --optimizer adamw --diffuser_type mlp1 --hidden_factor 64 --random_seed 100
```

### KuaiRec Data

```
python -u DreamRec.py --data ks --cuda 0 --timesteps 2000 --lr 0.0001 --beta_sche exp --optimizer adamw --diffuser_type mlp1 --hidden_factor 64 --random_seed 100
```

### Zhihu Data

```
python -u DreamRec.py --data zhihu --cuda 0 --timesteps 500 --lr 0.01 --beta_sche linear --optimizer adamw --diffuser_type mlp1 --hidden_factor 64 --random_seed 100
```
