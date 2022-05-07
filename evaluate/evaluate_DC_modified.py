import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import configs
from io_utils import parse_args
import time
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
use_gpu = torch.cuda.is_available()




def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
        #dist.append(torch.cdist(query, base_means[i], p=2))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov



def SEN(Y_test,Y_pred,n):
    
    sen = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1*100)
        
    return sen

def PRE(Y_test,Y_pred,n):
    
    pre = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:,i]) - tp
        if (tp + fp) == 0:
            pre1 = 0.0
        else:
            pre1 = tp / (tp + fp)
        pre.append(pre1*100)
        
    return pre

def SPE(Y_test,Y_pred,n):
    
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1*100)
    
    return spe

def ACC(Y_test,Y_pred,n):
    
    acc = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    #acc = np.sum(np.diag(con_mat))/np.sum(con_mat)*100
    #'''
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1*100)
    #'''
    return acc


if __name__ == '__main__':
    # ---- data loading
    params = parse_args('test')
    dataset = params.dataset_file
    n_shot = params.n_shot
    n_ways = params.n_way
    n_queries = 5
    n_runs = 600
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples


    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    #ndatas /= (np.linalg.norm(ndatas, axis=1).reshape(-1, 1) + 1e-10) # L2-norm
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs, n_samples)
    
    # ---- Base class statistics
    base_means = []
    base_cov = []

    base_features_path = "%s/checkpoints/%s/%s_%s/last/base_features.plk"%(params.save_dir, params.dataset, params.model, params.method)
    
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            beta = 0.5
            feature = np.power(feature[:, ] ,beta)
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)
    
    
    # ---- classification for each task
    acc_list = []
    sen_list = []
    spe_list = []
    pre_list = []
    f1_list = []

    print('Start classification for %d tasks...'%(n_runs))
    for i in tqdm(range(n_runs)):

        support_data = ndatas[i][:n_lsamples].numpy()
        
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- Tukey's transform
        beta = 0.5
        support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)
        
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(750/n_shot)
        for i in range(n_lsamples):
            mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=2)
            sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[i]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = np.concatenate([support_data, sampled_data])
        Y_aug = np.concatenate([support_label, sampled_label])
        
        #X_aug = torch.tensor(X_aug).cuda()
        #Y_aug = torch.tensor(Y_aug).cuda()
        # ---- train classifier
        classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)

        predicts = classifier.predict(query_data)
        
        cnf_matrix = confusion_matrix(query_label,predicts)
        #print(cnf_matrix)
        sen_ = np.mean(SEN(query_label, predicts, n_ways))
        spe_ = np.mean(SPE(query_label, predicts, n_ways))
        pre_ = np.mean(PRE(query_label, predicts, n_ways))
        acc_ = np.mean(ACC(query_label, predicts, n_ways))
        if pre_==0 or sen_ == 0:
            f1_ = 0.0
        else:
            f1_ = 2*pre_*sen_/(pre_+sen_)
        
        sen_list.append(sen_)
        spe_list.append(spe_)
        pre_list.append(pre_)
        f1_list.append(f1_)
        acc_list.append(acc_)

    mean_sen = float(np.mean(sen_list))
    sen_std = np.std(sen_list)
    mean_spe = float(np.mean(spe_list))
    spe_std = np.std(spe_list)
    mean_pre = float(np.mean(pre_list))
    pre_std = np.std(pre_list)
    mean_f1 = float(np.mean(f1_list))
    f1_std = np.std(f1_list)
    mean_acc = float(np.mean(acc_list))
    acc_std = np.std(acc_list)
    


    print('%s %d way %d shot  SEN : %4.2f%% +- %4.2f%%'%(dataset,n_ways,n_shot,mean_sen,1.96*sen_std/np.sqrt(n_runs)))
    print('%s %d way %d shot  SPE : %4.2f%% +- %4.2f%%'%(dataset,n_ways,n_shot,mean_spe,1.96*spe_std/np.sqrt(n_runs)))
    print('%s %d way %d shot  PRE : %4.2f%% +- %4.2f%%'%(dataset,n_ways,n_shot,mean_pre,1.96*pre_std/np.sqrt(n_runs)))
    print('%s %d way %d shot  F1-score : %4.2f%% +- %4.2f%%'%(dataset,n_ways,n_shot,mean_f1,1.96*f1_std/np.sqrt(n_runs)))
    print('%s %d way %d shot  ACC : %4.2f%% +- %4.2f%%'%(dataset,n_ways,n_shot,mean_acc,1.96*acc_std/np.sqrt(n_runs)))

    
    with open('./record/results.txt' , 'a+') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''
        #aug_str += '-adapted' if params.adaptation else ''
        exp_setting = '%s-%s-%s%s %sshot %sway_test' %(params.dataset, params.model, params.method, aug_str , params.n_shot , params.n_way)
        acc_str = '%s %d way %d shot  ACC : %4.2f%% +- %4.2f%%'%(dataset,n_ways,n_shot,mean_acc,1.96*acc_std/np.sqrt(n_runs))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )

