# Use with `bash ./unsup-parts/vessl_setup.sh`
apt-get update
apt-get -y install libgl1-mesa-glx

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source /root/.bashrc

cd unsup-parts
conda env create --file environment_a6000.yaml
conda activate unsup-parts

mkdir data
unzip /input/VOCdevkit_2010.zip -d ./data

mkdir outputs
unzip /input/entire-run4-sc50.zip -d ./outputs


wandb login $wandb_api_key

# How to run
# export pascal_class="['dog']"
# export lambda_sc=5
# export use_kl_distill="True"
# export wandb_api_key=""
# export exp_name=""
# python train_distill.py dataset_name=PP \\
#     lambda_sc=$lambda_sc \\
#     pascal_class=$pascal_class \\
#     use_kl_distill=$use_kl_distill \\
#     exp_name=$exp_name