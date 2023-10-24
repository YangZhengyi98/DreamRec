# DreamRec

This is the implementation of our NeurIPS 2023 paper:

> Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, and Xiangnan He. Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion. In NeurIPS 2023.

![framework](./fig/method.pdf "framework")

## Reproduce the results

### YooChoose Data

```
python -u DreamRec.py --data yc --timesteps 500 --lr 0.001 --beta_sche exp --w 2 --optimizer adamw --diffuser_type mlp1 --random_seed 100
```

### KuaiRec Data

```
python -u DreamRec.py --data ks --timesteps 2000 --lr 0.0001 --beta_sche cosine --w 2 --optimizer adamw --diffuser_type mlp1 --random_seed 100
```

### Zhihu Data

```
python -u DreamRec.py --data zhihu --timesteps 500 --lr 0.01 --beta_sche linear --w 4 --optimizer adamw --diffuser_type mlp1 --random_seed 100 
```
