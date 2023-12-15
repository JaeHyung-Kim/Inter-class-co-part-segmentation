# Inter-Class Co-Part Segmentation Matching Using Self-Supervision

#### Jaehyung Kim, Sanghyeon Son, Baekrok Shin
This project is implemented based on [original Unsup-parts repo](https://github.com/subhc/unsup-parts.)


### Environment Setup

```shell
cd unsup-parts
conda env create --file environment.yml
conda activate unsup-parts
```
The project uses Weights & Biases for visualization, please update `wandb_userid` in `train.py` to your username




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

#### To train a baseline Unsup-parts: 
1. Set the `pascal_class` to `["${class_you_want}"].`
2. Start training with the following command.
```shell
python train.py
```
Checkpoints and training log will be created under `outputs` and `wandb` folders each.

#### To train a teacher network:
1. Change the `model` to `DeepLab101_2branch` from `DeepLab50_2branch` in `config/config.yaml`.
2. Set the `pascal_class` to `["dog", "cat", "sheep"]` in `config/config.yaml`.
3. Start training with the following command.
```shell
python train.py
```

#### To train a student network:
1. Set the `model` to `DeepLab50_2branch` in `config/config.yaml`.
2. Set the `restore_teacher_from` to `${your_teacher_checkpoint}.pth"`.
3. Set the `pascal_class` to `["${class_you_want}"]`.
4. Start training with the following command.
```shell
python train_distill.py
```

### Evaluation:
1. Set the `model` to `DeepLab50_2branch` or `DeepLab101_2branch` in `config/config.yaml` by the model you want to evaluate.
2. Set the `restore_from` to `${checkpoint_to_eval}.pth"`.
3. Set the `pascal_class` to `["${class_you_want}"]`.
3. Run `eval.py` to evaluate the intra-class consistency. This will generate `copart_pred_{self.args.pascal_class[0]}.npy`
4. Once you've evaluated all ["dog", "cat", "sheep"] classes, run `overall_consistency` to evaluate inter-class consistency.


### Pretrained weights:
TODO
