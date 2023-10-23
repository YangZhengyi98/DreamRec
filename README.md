# DreamRec

This is the implementation of our NeurIPS 2023 paper:

> Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, and Xiangnan He. Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion. In NeurIPS 2023.

## Reprocuce the results

### YooChoose Data

```
python -u DreamRec.py --data yc --cuda 0 --timesteps 500 --lr 0.001 --beta_sche exp --w 2 --optimizer adamw --diffuser_type mlp1 --hidden_factor 64 --random_seed 100
```

### KuaiRec Data

```
python -u DreamRec.py --data ks --cuda 0 --timesteps 2000 --lr 0.0001 --beta_sche exp --w 2 --optimizer adamw --diffuser_type mlp1 --hidden_factor 64 --random_seed 100
```

### Zhihu Data

```
python -u A_A_SDRecv2cfg.py --data zhihuv4 --cuda 5 --timesteps 500 --lr 0.01 --beta_sche linear --w 4 --optimizer adamw --diffuser_type mlp1 --hidden_factor 64 --random_seed 100 
```
