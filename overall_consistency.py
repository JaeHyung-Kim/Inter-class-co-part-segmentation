from sklearn.metrics import normalized_mutual_info_score
from utils.utils import adjusted_rand_score_overflow
from itertools import combinations
import random
import numpy as np
import torch

copart_cat = np.load('copart_ioucat.npy')
copart_dog = np.load('copart_ioudog.npy')
copart_sheep = np.load('copart_iousheep.npy')

highest_iou_copart = np.concatenate((copart_cat, copart_dog, copart_sheep), axis=0)
highest_iou_copart = torch.Tensor(highest_iou_copart)

copart_pairs = list(combinations(highest_iou_copart, 2))
ari_list = []
nmi_list = []
pair_indices = np.arange(len(copart_pairs))
random.shuffle(pair_indices)
for pair_idx in pair_indices[:10000]:
    pair = copart_pairs[pair_idx]
    pair_ari = adjusted_rand_score_overflow(pair[0], pair[1])
    pair_nmi = normalized_mutual_info_score(pair[0], pair[1])
    ari_list.append(pair_ari)
    nmi_list.append(pair_nmi)

ari_list = np.array(ari_list)
nmi_list = np.array(nmi_list)

ari = np.average(ari_list)
nmi = np.average(nmi_list)

print(f"co-part ARI: {ari * 100: .2f}, co-part NMI: {nmi * 100: .2f}")