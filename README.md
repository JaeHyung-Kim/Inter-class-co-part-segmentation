# Inter-Class Co-Part Segmentation Matching Using Self-Supervision

#### Jaehyung Kim, Sanghyeon Son, Baekrok Shin
This project is implemented based on [original Unsup-parts repo](https://github.com/subhc/unsup-parts).


### Environment Setup
```shell
conda env create --file environment.yml
conda activate unsup-parts
```

### Data Preparation:

#### PASCAL-PART

1. Download [VOCdevkit_2010.zip](https://drive.google.com/file/d/1RdhW5K8Jk7cqh-CX4NZy4LdLHDH6Y3eZ/view?usp=sharing).
2. Create a directory named `data` with the following folder structure inside and extract the zip at the mentioned locations.
```shell
data
└── VOCdevkit_2010  # extract VOCdevkit_2010.zip here
    ├── Annotations_parts   
    ├── examples
    ├── VOC2010
    ├── demo.m
    ├── mat2map.m
    ├── part2ind.m
    └── VOClabelcolormap.m
```

### Training:

This project uses wandb. **Please make sure to associate wandb to your account before the training.** Otherwise, you will not be able to run any script.
```shell
wandb login

# Type your login key after.
```


#### To train a baseline Unsup-parts: 
1. Start training with the following command. `'dog'`, `'cat'`, `'sheep'` are available.
```shell
python train.py pascal_class="['dog']"
```
Checkpoints and training log will be created under `outputs` and `wandb` folders each.

#### To train a teacher network:
1. Start training with the following command. Only for the teacher model, we use `lambda_sc` to `50` and ResNet101. This command will use the entire classes.
```shell
python train.py lambda_sc=50 model=DeepLab101_2branch pascal_class="['dog', 'cat', 'sheep']"
```

#### To train a student network:
1. Start training with the following command. `'dog'`, `'cat'`, `'sheep'` are available. Pretrained teacher can be associated using `checkpoints/outputs/teacher/files/model_100000.pth`.
```shell
python train_distill.py pascal_class="['dog']" use_teacher_scheduling=True restore_teacher_from="${path_to_your_teacher_checkpoint}.pth"
```

### Evaluation:
1. Select `DeepLab50_2branch` or `DeepLab101_2branch` by the model you want to evaluate in `model=`.
2. Select checkpoint you want in `restore_from=`.
3. Select class you want to evaluate.
3. Run `eval.py` to evaluate the intra-class consistency. This will generate `copart_pred_${class}.npy`
```shell
python eval.py pascal_class="['dog']" restore_from="${path_to_your_teacher_checkpoint}.pth" model="${resnet50_or_101}$
```

4. Once you've evaluated all ["dog", "cat", "sheep"] classes, run `overall_consistency` to evaluate inter-class consistency.
```shell
python overall_consistency.py
```

### Pretrained weights:

Pretrained weights can be downloaded from [here](https://drive.google.com/file/d/1Kz8zkpnkzKTqv1iD9nhVZcp073xloZ-1/view?usp=sharing). Please make sure to associate them using `restore_teacher_from=` when distillation training and `restore_from=` when evaluation.
