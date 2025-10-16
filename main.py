import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
import models
import utils
from subutils import *
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import time

def fix_seed(rseed):
    random.seed(rseed)
    np.random.seed(rseed)
    os.environ["PYTHONHASHSEED"] = str(rseed)
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=32, help='The hidden dimensions used in the classifier')
parser.add_argument('--embed-time', type=int, default=128, help='The number of reference time points and the dimension of time embedding functions')
parser.add_argument('--num-heads', type=int, default=1, help='The number of heads for multi-head attention mechanism')
parser.add_argument('--alpha', type=float, default=0.1, help='The loss weight for temporal consistency regularization')
parser.add_argument('--beta', type=float, default=0.01, help='The loss weight for inter-variable consistency regularization')
parser.add_argument('--lr', type=float, default=0.01, help='The learning rate')
parser.add_argument('--learn-emb', action='store_true', help='Type of time embedding function (Sinusoidal or Learnable)')
parser.add_argument('--cl-type', type=str, default='both', choices=['both', 'instance', 'temporal'], help='Type of contrastive loss function')
parser.add_argument('--dataset', type=str, default='PAM', choices=['P12', 'P19', 'PAM'], help='The dataset name')
parser.add_argument('--withmissingratio', default=False, help='If True, missing ratio ranges from 0 to 0.5; If False, missing ratio=0')
parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample', 'time'], help='Experiments for demonstrating robustness to missing')
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')

args = parser.parse_args()

logger = utils.setup_logger(name='ATE', log_file='results/log/ATE_'+args.dataset+'.log')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def logspace(start, end, steps):
    values = torch.linspace(start, end, steps)
    log_values = torch.exp(values)
    log_values = (log_values - log_values.min()) / (log_values.max() - log_values.min())
    return log_values

def reverse_logspace(start, end, steps):
    log_values = torch.linspace(start, end, steps)
    reversed_values = torch.exp(log_values.flip(0))
    reversed_values = (reversed_values - reversed_values.min()) / (reversed_values.max() - reversed_values.min())
    reversed_values = (reversed_values - 1).abs()
    return reversed_values

def random_masking(irr_mask, mask_ratio):
    irr_mask = irr_mask
    temp = torch.full((irr_mask.size(0), irr_mask.size(1)), mask_ratio)
    temp = (1 - torch.bernoulli(temp))
    temp = temp.repeat_interleave(irr_mask.size(-1), dim=-1).view(irr_mask.size())
    mask = irr_mask.clone() * temp
    return mask.cuda()

dataset = args.dataset
logger.info('Dataset used: %s' % (dataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if dataset == 'P12':
    base_path = 'dataset/P12data'
elif dataset == 'P19':
    base_path = 'dataset/P19data'
elif dataset == 'PAM':
    base_path = 'dataset/PAMdata'

model_path = 'models/'

logger.info('Model: ATE')
logger.info('Hidden dim: %i' % (args.hidden))
logger.info('Embedding time: %i' % (args.embed_time))
logger.info('Num heads: %i' % (args.num_heads))
logger.info('Alpha: %f' % (args.alpha))
logger.info('Beta: %f' % (args.beta))
logger.info('Learning embedding: %s' % (args.learn_emb))
logger.info('CL type: %s' % (args.cl_type))
logger.info('Learning rate: %f' % (args.lr))


split = 'random'
reverse = False
feature_removal_level = args.feature_removal_level  # 'set', 'sample', time

if args.withmissingratio == True:
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
else:
    missing_ratios = [0]

for missing_ratio in missing_ratios:
    num_epochs = 20
    learning_rate = args.lr

    if dataset == 'P12':
        d_static = 9
        d_inp = 36
        static_info = 0
    elif dataset == 'P19':
        d_static = 6
        d_inp = 34
        static_info = 0
    elif dataset == 'PAM':
        d_static = 0
        d_inp = 17
        static_info = 0

    if dataset == 'P12':
        n_classes = 2
    elif dataset == 'P19':
        n_classes = 2
    elif dataset == 'PAM':
        n_classes = 8

    n_runs = 1
    n_splits = 5
    subset = False

    acc_arr = np.zeros((n_splits, n_runs))
    auprc_arr = np.zeros((n_splits, n_runs))
    auroc_arr = np.zeros((n_splits, n_runs))
    precision_arr = np.zeros((n_splits, n_runs))
    recall_arr = np.zeros((n_splits, n_runs))
    F1_arr = np.zeros((n_splits, n_runs))

    for k in range(n_splits):
        split_idx = k + 1
        print('Split id: %d' % split_idx)

        if dataset == 'P12':
            if subset == True:
                split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'P19':
            split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        elif dataset == 'PAM':
            split_path = '/splits/PAMAP2_split_' + str(split_idx) + '.npy'

        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split,
                                                                  reverse=reverse, dataset=dataset,
                                                                  predictive_label='mortality')

        if dataset == 'P12' or dataset == 'P19':
            T, F = Ptrain[0]['arr'].shape
            D = len(Ptrain[0]['extended_static'])

            Ptrain_tensor = np.zeros((len(Ptrain), T, F))
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            for i in range(len(Ptrain)):
                Ptrain_tensor[i] = Ptrain[i]['arr']
                Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            mf, stdf = getStats(Ptrain_tensor)
            ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf, stdf, ms, ss)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms, ss)
            
        elif dataset == 'PAM':
            T, F = Ptrain[0].shape
            D = 1

            Ptrain_tensor = Ptrain
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            mf, stdf = getStats(Ptrain)
            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)
            
        # remove part of variables in validation and test set
        if missing_ratio > 0:
            num_all_features = int(Pval_tensor.shape[2] / 2)
            num_missing_features = round(missing_ratio * num_all_features)
            if feature_removal_level == 'sample':
                for i, patient in enumerate(Pval_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)  # values
                    Pval_tensor[i] = patient
                for i, patient in enumerate(Ptest_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)  # values
                    Ptest_tensor[i] = patient
            elif feature_removal_level == 'set':
                density_score_indices = np.load('saved/IG_density_scores_' + dataset + '.npy', allow_pickle=True)[:, 0]
                idx = density_score_indices[:num_missing_features].astype(int)
                Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)  # values
                Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)  # values

        Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
        Pval_tensor = Pval_tensor.permute(1, 0, 2)
        Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

        if missing_ratio > 0 and feature_removal_level == 'time':
            test_mask = random_masking(Ptest_tensor[:, :, d_inp:], missing_ratio)
            Ptest_tensor[:, :, d_inp:] = test_mask
        
        Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
        Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
        Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

        for m in range(n_runs):
            logger.info('- - Run %d - -' % (m + 1))
            if dataset == 'P12' or dataset == 'P19':
                model = models.ATE(d_inp, d_static, args.hidden, embed_time=args.embed_time, learn_emb=args.learn_emb,
                                      num_heads=args.num_heads, device=device, n_classes=n_classes, static=False).to(device)
            elif dataset == 'PAM':
                model = models.ATE(d_inp, d_static, args.hidden, embed_time=args.embed_time, learn_emb=args.learn_emb,
                                      num_heads=args.num_heads, device=device, n_classes=n_classes, static=False).to(device)
            logger.info('Number of parameters: %i', count_parameters(model))
            model = model.cuda()

            criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
            bce_criterion = nn.BCELoss().cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                                   patience=1, threshold=0.0001, threshold_mode='rel',
                                                                   cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

            idx_0 = np.where(ytrain == 0)[0]
            idx_1 = np.where(ytrain == 1)[0]

            if dataset == 'P12' or dataset == 'P19':
                strategy = 2
            elif dataset == 'PAM':
                strategy = 3
                
            n0, n1 = len(idx_0), len(idx_1)
            expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
            expanded_n1 = len(expanded_idx_1)

            if args.num_heads == 4:
                batch_size = 64
            else:
                batch_size = 128
            if strategy == 1:
                n_batches = 10
            elif strategy == 2:
                K0 = n0 // int(batch_size / 2)
                K1 = expanded_n1 // int(batch_size / 2)
                n_batches = np.min([K0, K1])
            elif strategy == 3:
                n_batches = 30
            
            best_aupr_val = best_auc_val = 0.0
            logger.info('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (num_epochs, n_batches, num_epochs * n_batches))
            times = []
            
            for epoch in range(num_epochs):
                model.train()
                start = time.time()
                if strategy == 2:
                    np.random.shuffle(expanded_idx_1)
                    I1 = expanded_idx_1
                    np.random.shuffle(idx_0)
                    I0 = idx_0

                for n in range(n_batches):
                    if strategy == 1:
                        idx = random_sample(idx_0, idx_1, batch_size)
                    elif strategy == 2:
                        """In each batch=128, 64 positive samples, 64 negative samples"""
                        idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                    elif strategy == 3:
                        idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                               Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                    elif dataset == 'PAM':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                               None, ytrain_tensor[idx].cuda()
                        
                    out, out1, pred_y, sx, sout, query = evaluate_ATE(model, P, Ptime, None, static=static_info)
                    optimizer.zero_grad()
                    
                    # Classification loss function
                    ce_loss = criterion(pred_y, y)

                    # Temporal consistency regularization
                    if args.cl_type == 'both':
                        cl_loss = utils.temporal_contrastive_loss(out, out1) + utils.instance_contrastive_loss(out, out1)
                    elif args.cl_type == 'instance':
                        cl_loss = utils.instance_contrastive_loss(out, out1)
                    elif args.cl_type == 'temporal':
                        cl_loss = utils.temporal_contrastive_loss(out, out1)

                    # Inter-variable consistency regularization
                    bce_loss = bce_criterion(sout, sx)

                    # Optimization
                    loss = ce_loss + args.alpha * cl_loss + args.beta * bce_loss
                    loss.backward()
                    optimizer.step()

                end = time.time()
                times.append(end - start)

                """Validation"""
                model.eval()
                with torch.no_grad():
                    val_out, val_out1, val_pred_y, val_sx, val_sout, _ = evaluate_ATE_batch(model, Pval_tensor, Pval_time_tensor, None, static=static_info, n_classes=n_classes)
                    val_pred_y = torch.squeeze(torch.sigmoid(val_pred_y))
                    val_pred_y = val_pred_y.detach().cpu().numpy()

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                        auc_val = roc_auc_score(yval, val_pred_y[:, 1])
                        aupr_val = average_precision_score(yval, val_pred_y[:, 1])
                    elif dataset == 'PAM':
                        auc_val = roc_auc_score(one_hot(yval), val_pred_y)
                        aupr_val = average_precision_score(one_hot(yval), val_pred_y)

                    scheduler.step(auc_val)
                    if auc_val > best_auc_val:
                        best_auc_val = auc_val
                        torch.save(model.state_dict(), model_path + 'ATE_model_' + str(split_idx) + '.pt')
                
            logger.info('Total Time elapsed: %.3f' % (sum(times) / num_epochs))  

            """Testing"""
            model.load_state_dict(torch.load(model_path + 'ATE_model_' + str(split_idx) + '.pt'))
            model.eval()

            with torch.no_grad():
                _, _, test_pred_y, _, _, query = evaluate_ATE_batch(model, Ptest_tensor, Ptest_time_tensor, None, static=static_info, n_classes=n_classes)
                test_pred_y = torch.squeeze(torch.sigmoid(test_pred_y))
                test_pred_y = test_pred_y.detach().cpu().numpy()
                ypred = np.argmax(test_pred_y, axis=1)

                acc = np.sum(ytest.ravel() == ypred.ravel()) / ytest.shape[0]

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    auc = roc_auc_score(ytest, test_pred_y[:, 1])
                    aupr = average_precision_score(ytest, test_pred_y[:, 1])
                    precision = precision_score(ytest, ypred)
                    recall = recall_score(ytest, ypred)
                    F1 = f1_score(ytest, ypred)
                elif dataset == 'PAM':
                    auc = roc_auc_score(one_hot(ytest), test_pred_y)
                    aupr = average_precision_score(one_hot(ytest), test_pred_y)
                    precision = precision_score(ytest, ypred, average='macro', )
                    recall = recall_score(ytest, ypred, average='macro', )
                    F1 = f1_score(ytest, ypred, average='macro', )
                
                logger.info('Testing: Precision = %.3f | Recall = %.3f | F1 = %.3f' % (precision * 100, recall * 100, F1 * 100))
                logger.info('Testing: AUROC = %.3f | AUPRC = %.3f | Accuracy = %.3f' % (auc * 100, aupr * 100, acc * 100))
                # print('classification report', classification_report(ytest, ypred))
                # print(confusion_matrix(ytest, ypred, labels=list(range(n_classes))))

                learned_query = query.cpu().detach().numpy()
                learned_query = (learned_query - learned_query.min()) / (learned_query.max() - learned_query.min())
                plt.figure(figsize=(7, 3))
                plt.plot(learned_query, np.zeros_like(learned_query) + 0.4, 'ro', markersize=3, label='Adaptive Interval (ours)')
                plt.plot(np.linspace(0, 1., 128), np.zeros_like(learned_query) + 0.3, 'gx', markersize=3, label='Regular Interval')
                plt.plot(logspace(0, 1., 128), np.zeros_like(learned_query) + 0.2, 'bx', markersize=3, label='Log Interval')
                plt.plot(reverse_logspace(0, 1., 128), np.zeros_like(learned_query) + 0.1, 'mx', markersize=3, label='Reverse Log Interval')
                plt.xticks([], [])
                plt.yticks(np.arange(0, 0.5, 0.1), ['', 'Dense', 'Sparse', 'Regular', 'ATE'], fontsize=15)
                plt.ylim(0.05,0.45)
                plt.tight_layout()
                plt.savefig(f'results/fig/{dataset}_RUN_{k}_selected_reference_points.png', dpi=300)
                plt.close() 

            # store
            acc_arr[k, m] = acc * 100
            auprc_arr[k, m] = aupr * 100
            auroc_arr[k, m] = auc * 100
            precision_arr[k, m] = precision * 100
            recall_arr[k, m] = recall * 100
            F1_arr[k, m] = F1 * 100

    # pick best performer for each split based on max AUPRC
    idx_max = np.argmax(auprc_arr, axis=1)
    acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
    auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
    auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]
    precision_vec = [precision_arr[k, idx_max[k]] for k in range(n_splits)]
    recall_vec = [recall_arr[k, idx_max[k]] for k in range(n_splits)]
    F1_vec = [F1_arr[k, idx_max[k]] for k in range(n_splits)]

    # display mean and standard deviation
    mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
    mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
    mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
    logger.info('------------------------------------------')
    logger.info('Accuracy = %.3f +/- %.3f' % (mean_acc, std_acc))
    logger.info('AUPRC    = %.3f +/- %.3f' % (mean_auprc, std_auprc))
    logger.info('AUROC    = %.3f +/- %.3f' % (mean_auroc, std_auroc))
    mean_precision, std_precision = np.mean(precision_vec), np.std(precision_vec)
    mean_recall, std_recall = np.mean(recall_vec), np.std(recall_vec)
    mean_F1, std_F1 = np.mean(F1_vec), np.std(F1_vec)
    logger.info('Precision = %.3f +/- %.3f' % (mean_precision, std_precision))
    logger.info('Recall    = %.3f +/- %.3f' % (mean_recall, std_recall))
    logger.info('F1        = %.3f +/- %.3f' % (mean_F1, std_F1))