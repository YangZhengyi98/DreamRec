# Diff4Rec

## Update the data, code and training log of SASRec on YooChoose dataset. -- 2023.8.29

## Update the experimental results on YooChoose data under the random split setting. -- 2023.8.30

We have just obtained the preliminary results using the random data split setting with the same data provided in work [1]. The initial results are outlined below (the results of SASRec and SASRec-PRL are directly extracted  from work [1]):

 |  | YooChoose (random split)| - |
 | :----: | :----: | :----: |
 |  | HR@20 | NDCG@20 |
 | SASRec | 0.6329 | 0.3558 |
 | SASRec-PRL | 0.6893 | 0.4013 |
 | Diff4Rec | 0.6920 | 0.4506 |


We can observe that Diff4Rec outperforms SASRec and more recent work SASRec-PRL [1], which suggests Diff4Rec maintains its efficacy  under the random split setting.

 [1] Rethinking reinforcement learning for recommendation: A prompt perspective. In SIGIR, 2022: 1347-1357.
