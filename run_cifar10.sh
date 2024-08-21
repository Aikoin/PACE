#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/source.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/norm.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/tent.yaml
CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cotta.yaml


