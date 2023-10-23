import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from Modules_ori import *


logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='yc',
                        help='yc, ks, zhihu')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--model_name', type=str, default='SASRec_bce',
                        help='model name.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-6,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    return parser.parse_args()

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output



def evaluate(model, test_data, device):
    eval_data=pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    evaluated=0
    total_clicks=1.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]

    seq, len_seq, target = list(eval_data['seq']), list(eval_data['len_seq']), list(eval_data['next'])

    num_total = len(seq)

    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction = model.forward_eval(states, np.array(len_seq_b))
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit(sorted_list2,topk,target_b,hit_purchase,ndcg_purchase)

        total_purchase+=batch_size
    # while evaluated<len(eval_ids):
    #     states, len_states, actions, rewards = [], [], [], []
    #     for i in range(batch):
    #         id=eval_ids[evaluated]
    #         group=groups.get_group(id)
    #         history=[]
    #         for index, row in group.iterrows():
    #             state=list(history)
    #             state = [int(i) for i in state]
    #             len_states.append(seq_size if len(state)>=seq_size else 1 if len(state)==0 else len(state))
    #             state=pad_history(state,seq_size,item_num)
    #             states.append(state)
    #             action=row['item_id']
    #             try:
    #                 is_buy=row['t_read']
    #             except:
    #                 is_buy=row['time']
    #             reward = 1 if is_buy >0 else 0
    #             if is_buy>0:
    #                 total_purchase+=1.0
    #             else:
    #                 total_clicks+=1.0
    #             actions.append(action)
    #             rewards.append(reward)
    #             history.append(row['item_id'])
    #         evaluated+=1
    #         if evaluated >= len(eval_ids):
    #             break

    #     states = np.array(states)
    #     states = torch.LongTensor(states)
    #     states = states.to(device)

    #     prediction = model.predict(states, np.array(len_states), diff)
    #     # print(prediction)
    #     _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
    #     topK = topK.cpu().detach().numpy()
    #     # prediction = prediction.cpu()
    #     # prediction = prediction.detach().numpy()
    #     # print(prediction)
    #     # prediction=sess.run(GRUnet.output, feed_dict={GRUnet.inputs: states,GRUnet.len_state:len_states,GRUnet.keep_prob:1.0})
    #     # sorted_list=np.argsort(prediction)
    #     sorted_list2=np.flip(topK,axis=1)
    #     calculate_hit(sorted_list2,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)
    print('#############################################################')
    # logging.info('#############################################################')
    # print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    # logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    hr_list = []
    ndcg_list = []
    print('hr@{}\tndcg@{}\thr@{}\tndcg@{}\thr@{}\tndcg@{}'.format(topk[0], topk[0], topk[1], topk[1], topk[2], topk[2]))
    # logging.info('#############################################################')
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        # print(hr_purchase)
        # print(ng_purchase)

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])
        # ndcg_list.append(ng_purchase)

        if i == 1:
            hr_20 = hr_purchase

    print('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    print('{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    # logging.info('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    # logging.info('{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    print('#############################################################')
    # logging.info('#############################################################')

    return hr_20


def calcu_propensity_score(buffer):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.5)
    return ps



if __name__ == '__main__':

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # logging.basicConfig(filename="./log/{}/{}_{}_lr{}_decay{}_dro{}_gamma{}".format(args.data + '_final2', Time.strftime("%m-%d %H:%M:%S", Time.localtime()), args.model_name, args.lr, args.l2_decay, args.dro_reg, args.gamma))
    # Network parameters

    data_directory = './data/' + args.data
    # data_directory = './data/' + args.data
    # data_directory = '../' + args.data + '/data'
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk=[10, 20, 50]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = SASRec(args.hidden_factor,item_num, seq_size, args.dropout_rate, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    ps = calcu_propensity_score(train_data)
    ps = torch.tensor(ps)
    ps = ps.to(device)

    total_step=0
    hr_max = 0
    best_epoch = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    for i in range(args.epoch):
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target=list(batch['next'].values())

            target_neg = []
            for index in range(args.batch_size):
                neg=np.random.randint(item_num)
                while neg==target[index]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            target_neg = torch.LongTensor(target_neg)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = target_neg.to(device)

            model_output = model.forward(seq, len_seq)


            target = target.view(args.batch_size, 1)
            target_neg = target_neg.view(args.batch_size, 1)

            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)

            pos_labels = torch.ones((args.batch_size, 1))
            neg_labels = torch.zeros((args.batch_size, 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(device)

            loss = bce_loss(scores, labels)


            pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
            pos_scores_dro = torch.squeeze(pos_scores_dro)
            pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
            pos_loss_dro = torch.squeeze(pos_loss_dro)

            inner_dro = torch.sum(torch.exp(torch.clamp(torch.mul(model_output * model_output, ps) / args.beta, max=1e2)), 1) - torch.exp(torch.clamp(pos_scores_dro / args.beta, max=1e2)) + torch.exp(torch.clamp(pos_loss_dro / args.beta, max=1e2)) 

            # A = torch.sum(torch.exp(torch.mul(model_output * model_output, ps)), 1)
            # B = torch.exp(pos_scores_dro)
            # C = torch.exp(pos_loss_dro) 
            # print(A.shape, B.shape, C.shape)

            loss_dro = torch.log(inner_dro + 1e-24)
            if args.alpha == 0.0:
                loss_all = loss
            else:
                loss_all = loss + args.alpha * torch.mean(loss_dro)
            loss_all.backward()
            optimizer.step()

            if True:

                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth step is: %f" % (total_step, loss_all))
                    # logging.info("the loss in %dth step is: %f" % (total_step, loss_all))

                if total_step % 2000 == 0:
                        # print('VAL:')
                        # logging.info('VAL:')
                        # hr_20 = evaluate(model, 'val_sessions_pos.df', device)
                        print('VAL PHRASE:')
                        # logging.info('VAL PHRASE:')
                        hr_20 = evaluate(model, 'val_data.df', device)
                        print('TEST PHRASE:')
                        # logging.info('TEST PHRASE:')
                        _ = evaluate(model, 'test_data.df', device)
                        # print('TEST PHRASE3:')
                        # logging.info('TEST PHRASE3:')
                        # _ = evaluate(model, 'test_sessions3_pos.df', device)

                        if hr_20 > hr_max:

                            hr_max = hr_20
                            best_epoch = total_step
                        
                        print('BEST EPOCH:{}'.format(best_epoch))
                        # logging.info('BEST EPOCH:{}'.format(best_epoch))
                        all_item_emb = model.item_embeddings.weight.cpu().detach().numpy()
                        np.save('./visual_sasrec_{}/all_item_arr_{}.npy'.format(args.data, total_step), all_item_emb)




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

