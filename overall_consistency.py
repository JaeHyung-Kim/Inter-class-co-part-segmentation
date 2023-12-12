from sklearn.metrics import normalized_mutual_info_score
from utils.utils import adjusted_rand_score_overflow
from itertools import combinations
import random
import numpy as np
import torch

pred_cat = np.load('copart_pred_cat.npy') 
mask_cat = np.load('copart_mask_cat.npy')
num_cat  = pred_cat.shape[0]

pred_dog = np.load('copart_pred_dog.npy') 
mask_dog = np.load('copart_mask_dog.npy')
num_dog  = pred_dog.shape[0]

pred_sheep = np.load('copart_pred_sheep.npy') 
mask_sheep = np.load('copart_mask_sheep.npy')
num_sheep  = pred_sheep.shape[0]


def consistency_between_two_classes(pred_a, mask_a, pred_b, mask_b, num_samples=1000):
    num_a = pred_a.shape[0]
    num_b = pred_b.shape[0]
    # cat - dog
    ari = []
    nmi = []
    while len(ari)<num_samples:
        a_idx = np.random.randint(0, num_a)
        b_idx = np.random.randint(0, num_b)
        valid_copart = np.logical_and(mask_a[a_idx], mask_b[b_idx])
        if np.sum(valid_copart) >= 8:
        
            a_copart = pred_a[a_idx]
            b_copart = pred_b[b_idx]
            
            a_copart = a_copart[valid_copart != 0]
            b_copart = b_copart[valid_copart != 0]
            
            pair_ari = adjusted_rand_score_overflow(a_copart, b_copart)
            pair_nmi = normalized_mutual_info_score(a_copart, b_copart)
            ari.append(pair_ari)
            nmi.append(pair_nmi)
            
    ari = np.array(ari)
    nmi = np.array(nmi)
    
    return np.mean(ari), np.mean(nmi)
    
ari_cd, nmi_cd = consistency_between_two_classes(pred_cat, mask_cat, pred_dog, mask_dog)
ari_ds, nmi_ds = consistency_between_two_classes(pred_dog, mask_dog, pred_sheep, mask_sheep)
ari_sc, nmi_sc = consistency_between_two_classes(pred_sheep, mask_sheep, pred_cat, mask_cat)

print(f"co-part ARI: {(ari_cd+ari_ds+ari_sc) / 3. * 100: .2f}, co-part NMI: {(nmi_cd+nmi_ds+nmi_sc) / 3. * 100: .2f}")