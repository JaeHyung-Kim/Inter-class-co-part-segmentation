from sklearn.metrics import normalized_mutual_info_score
from utils.utils import adjusted_rand_score_overflow
from itertools import combinations
import random
import numpy as np
import torch
import os
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt


def consistency_between_two_classes(pred_a, mask_a, pred_b, mask_b, threshold, num_samples=10000):
    num_a = pred_a.shape[0]
    num_b = pred_b.shape[0]
    # cat - dog
    ari = []
    nmi = []

    a_stat = np.unique(np.sum(mask_a, axis=-1), return_counts=True)
    a_stat = {i: v for i, v in zip(*a_stat)}
    b_stat = np.unique(np.sum(mask_b, axis=-1), return_counts=True)
    b_stat = {i: v for i, v in zip(*b_stat)}
    for i in range(16):
        if i not in a_stat.keys():
            a_stat[i] = 0
        if i not in b_stat.keys():
            b_stat[i] = 0
            
    pair_stat = {i: 0 for i in range(16)}

    while len(ari)<num_samples:
        
        a_idx = np.random.randint(0, num_a)
        b_idx = np.random.randint(0, num_b)
        valid_copart = np.logical_and(mask_a[a_idx], mask_b[b_idx])

        pair_stat[np.sum(valid_copart)] += 1


        if np.sum(valid_copart) >= threshold:
        
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

    # fig, axes = plt.subplots(1,3, figsize=(14, 5))
    # axes[0].set_title("cat")
    # axes[0].bar(a_stat.keys(), a_stat.values())
    # axes[0].vlines(x=8, ymin=0, ymax=np.max(list(a_stat.values())), color="orange")
    # axes[1].set_title("dog")
    # axes[1].bar(b_stat.keys(), b_stat.values())
    # axes[1].vlines(x=8, ymin=0, ymax=np.max(list(b_stat.values())), color="orange")
    # axes[2].set_title("pair")
    # axes[2].bar(pair_stat.keys(), pair_stat.values())
    # axes[2].vlines(x=8, ymin=0, ymax=np.max(list(pair_stat.values())), color="orange")
    # fig.show()
    
    # print(a_stat)
    # print(b_stat)
    # print(pair_stat)
    
    return np.mean(ari), np.mean(nmi)
    



splits = [
    "_copart/baseline/",
    "_copart/teacher/",
    "_copart/distill",
    "_copart/distill_schedule"
]

keys = [
    "baseline_total_ari",
    "baseline_total_nmi",
    "baseline_cd_ari",
    "baseline_cd_nmi",
    "baseline_ds_ari",
    "baseline_ds_nmi",
    "baseline_sc_ari",
    "baseline_sc_nmi",
    "teacher_total_ari",
    "teacher_total_nmi",
    "teacher_cd_ari",
    "teacher_cd_nmi",
    "teacher_ds_ari",
    "teacher_ds_nmi",
    "teacher_sc_ari",
    "teacher_sc_nmi",
    "distill_total_ari",
    "distill_total_nmi",
    "distill_cd_ari",
    "distill_cd_nmi",
    "distill_ds_ari",
    "distill_ds_nmi",
    "distill_sc_ari",
    "distill_sc_nmi",
    "distill_schedule_total_ari",
    "distill_schedule_total_nmi",
    "distill_schedule_cd_ari",
    "distill_schedule_cd_nmi",
    "distill_schedule_ds_ari",
    "distill_schedule_ds_nmi",
    "distill_schedule_sc_ari",
    "distill_schedule_sc_nmi",
]

total_stat = {k: [] for k in keys}
for t in range(1, 9):
    t_stat = {}
    for i, split in tqdm(enumerate(splits)):
        pred_cat = np.load(os.path.join(split, 'copart_pred_cat.npy'))
        mask_cat = np.load(os.path.join(split, 'copart_mask_cat.npy'))
        num_cat  = pred_cat.shape[0]

        pred_dog = np.load(os.path.join(split, 'copart_pred_dog.npy'))
        mask_dog = np.load(os.path.join(split, 'copart_mask_dog.npy'))
        num_dog  = pred_dog.shape[0]

        pred_sheep = np.load(os.path.join(split, 'copart_pred_sheep.npy'))
        mask_sheep = np.load(os.path.join(split, 'copart_mask_sheep.npy'))
        num_sheep  = pred_sheep.shape[0]


        ari_cd, nmi_cd = consistency_between_two_classes(pred_cat, mask_cat, pred_dog, mask_dog, t)
        ari_ds, nmi_ds = consistency_between_two_classes(pred_dog, mask_dog, pred_sheep, mask_sheep, t)
        ari_sc, nmi_sc = consistency_between_two_classes(pred_sheep, mask_sheep, pred_cat, mask_cat, t)
        ari_totat, nmi_total = (ari_cd+ari_ds+ari_sc) / 3. * 100, (nmi_cd+nmi_ds+nmi_sc) / 3. * 100

        print(f"co-part ARI: {ari_totat: .2f}, co-part NMI: {nmi_total: .2f}")
        print(f"(cd) co-part ARI: {(ari_cd) * 100: .2f}, co-part NMI: {(nmi_cd) * 100: .2f}")
        print(f"(ds) co-part ARI: {(ari_ds) * 100: .2f}, co-part NMI: {(nmi_ds) * 100: .2f}")
        print(f"(sc) co-part ARI: {(ari_sc) * 100: .2f}, co-part NMI: {(nmi_sc) * 100: .2f}")

        t_stat[keys[8*i]] = ari_totat
        t_stat[keys[8*i + 1]] = nmi_total
        t_stat[keys[8*i + 2]] = ari_cd
        t_stat[keys[8*i + 3]] = nmi_cd
        t_stat[keys[8*i + 4]] = ari_ds
        t_stat[keys[8*i + 5]] = nmi_ds
        t_stat[keys[8*i + 6]] = ari_sc
        t_stat[keys[8*i + 7]] = nmi_sc

    for k in total_stat.keys():
        total_stat[k].append(t_stat[k])

df = pd.DataFrame(total_stat)
df.to_csv("dataframe.csv", header=True, index=True)




# def consistency_between_two_classes(pred_a, mask_a, pred_b, mask_b, num_samples=10000):
#     num_a = pred_a.shape[0]
#     num_b = pred_b.shape[0]
    
#     a_copart_list = []
#     b_copart_list = []
#     while len(a_copart_list)<num_samples:
#         a_idx = np.random.randint(0, num_a)
#         b_idx = np.random.randint(0, num_b)
#         valid_copart = np.logical_and(mask_a[a_idx], mask_b[b_idx])
#         if np.sum(valid_copart) >= 8:
        
#             a_copart = pred_a[a_idx]
#             b_copart = pred_b[b_idx]
            
#             a_copart = a_copart[valid_copart != 0]
#             b_copart = b_copart[valid_copart != 0]

#             a_copart_list.append(a_copart)
#             b_copart_list.append(b_copart)
            
#     a_copart_list = np.concatenate(a_copart_list)
#     b_copart_list = np.concatenate(b_copart_list)
    
#     pair_ari = adjusted_rand_score_overflow(a_copart_list, b_copart_list)
#     pair_nmi = normalized_mutual_info_score(a_copart_list, b_copart_list)

#     return pair_ari, pair_nmi
