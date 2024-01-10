## Anaconda Envirnoment
```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install scipy
    pip install matplotlib
    pip install torchsummary
    pip install tqdm (optional)


```

## Requirements
```
    python3 -m pip install git+https://github.com/ildoonet/pytorch-randaugment   (For Augmentation in Strong Augmentation!)
```

```
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```


## CUDA TOOLKIT 11.8[https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local]
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb

sudo cp /var/cuda-repo-ubuntu1804-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update-11-8

sudo apt-get -y install cuda-11-8
```
## Check
```
nvcc --version
```


## CUDNN 8.6.0 []
```
 tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
 sudo cp cuda/include/cudnn* /usr/local/cuda-11.8/include
 sudo cp cuda/lib/libcudnn* /usr/local/cuda-11.8/lib64
 sudo chmod a+r /usr/local/cuda-11.8/include/cudnn.h /usr/local/cuda-11.8/lib64/libcudnn*
```

## Check
```
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## Args 설명

## 실행 방법
### fixmatch (cifar10)
```
 nohup python main.py --training-method fixmatch --optim Adam --epoch 500 --val-iteration 500 --seed 0 --dataset cifar10 --num_max 1000 --label_ratio=20 --imb_ratio 100 --gpu 3 > fixmatch_cifar10.out &
```
### fixmatch (cifar100)
```
 nohup python main.py --training-method fixmatch --optim Adam --epoch 500 --val-iteration 500 --seed 0 --dataset cifar100 --num_max 200 --label_ratio=40 --imb_ratio 20 --gpu 0 --imbalancetype long > fixmatch_cifar100_4.out &
```
fixmatch+ABC
### fixmatch+ABC (cifar10)
```
 nohup python main.py --training-method fixmatchABC --optim Adam --epoch 500 --val-iteration 500 --seed 0 --dataset cifar10 --num_max 1000 --label_ratio=20 --imb_ratio 100 --gpu 3 > ./log/fixmatchABC.out &
```
### fixmatch+ABC (cifar100)
```
 nohup python main.py --training-method fixmatchABC --optim SGD --epoch 500 --val-iteration 500 --seed 0 --dataset cifar100 --num_max 200 --label_ratio=40 --imb_ratio 20 --imbalancetype step --gpu 1 > fixmatchABC_cifar100_1.out &
```

### fixmatch+DARP (cifar10)
```
nohup python main.py --training-method fixmatchDARP --optim SGD --epoch 500 --val-iteration 500 --seed 0  --dataset cifar10 --num_max 1000 --label_ratio 20 --imb_ratio 100 --gpu 3 --imbalancetype long --darp --est --alpha 2 --warm 200 > fixmatchDARP_cifar10.out &
```

### fixmatch+DARP (cifar100)
```
nohup python main.py --training-method fixmatchDARP  --epoch 500 --val-iteration 500 --seed 0  --dataset cifar100 --num_max 200 --label_ratio 40 --imb_ratio 20 --gpu 3 --imbalancetype step --darp --est --alpha 2 --warm 200 > fixmatchDARP_cifar100_1.out &
```

```
nohup python main.py --training-method fixmatchCCSSL  --epoch 500 --val-iteration 500 --seed 2 --dataset cifar10 --num_max 1000 --label_ratio 20 --imb_ratio 100 --gpu 0 --imbalancetype long --low_dim 64 --contrast_with_softlabel True --lambda_contrast 1. --contrastive_left_out False --contrastive_with_thresh 0.9 --temperature 0.07 --DA False > fixmatchCCSSL_cifar10_1.out &
```
### fixmatch+CCSSL (cifar100)
```
nohup python main.py --training-method fixmatchCCSSL  --epoch 500 --val-iteration 500 --seed 0  --dataset cifar100 --num_max 200 --label_ratio 40 --imb_ratio 20 --gpu 3 --imbalancetype step --low_dim 64 --contrast_with_softlabel True --lambda_contrast 1. --contrastive_left_out False --contrastive_with_thresh 0.9 --temperature 0.07 --DA False > fixmatchCCSSL_cifar100_3.out &
```



nohup python main.py --training-method fixmatch --optim SGD --epoch 500 --val-iteration 500 --seed 0 --dataset cifar100 --num_max 200 --label_ratio=40 --imb_ratio 20 --gpu 0 --imbalancetype step --train-strong True > fixmatch_cifar100.out &

nohup python main.py --training-method fixmatchABC --optim SGD --epoch 500 --val-iteration 500 --seed 0 --dataset cifar100 --num_max 200 --label_ratio=40 --imb_ratio 20 --imbalancetype step --gpu 1 --train-strong True > fixmatchABC_cifar100.out &

nohup python main.py --training-method fixmatchDARP  --optim SGD --epoch 500 --val-iteration 500 --seed 0  --dataset cifar100 --num_max 200 --label_ratio 40 --imb_ratio 20 --gpu 3 --imbalancetype step --darp --est --alpha 2 --warm 200 --train-strong True > fixmatchDARP_cifar100.out &


### fixmatch + ABC + proposed-1 (cifar10)
```
python main.py --training-method train_fixmatch_withABC_contrastive_labeled  --epoch 500 --val-iteration 500 --seed 0 --dataset cifar10 --num_max 1000 --label_ratio 20 --imb_ratio 100 --gpu 0 --imbalancetype long --low-dim 64 --temperature 0.07 --DA False --using-mask True --selfcon-pos False,False --contrastive-supervised True --lr 0.002 --singleView False1 --train-strong True --lambda-Scontrast .1 --mu 1 --contrastive-unlabeled supervised --contrastive-labeled supervised --eval ema
```

### fixmatch + ABC + proposed-1 (cifar100)
```
python main.py --training-method train_fixmatch_withABC_contrastive_labeled  --epoch 500 --val-iteration 500 --seed 0 --dataset cifar100 --num_max 200 --label_ratio 40 --imb_ratio 20 --gpu 0 --imbalancetype long --low-dim 64 --temperature 0.07 --DA False --using-mask True --selfcon-pos False,False --contrastive-supervised True --lr 0.002 --singleView False1 --train-strong True --lambda-Scontrast .1 --mu 1 --contrastive-unlabeled supervised --contrastive-labeled supervised --eval ema
```


### fixmatch + ABC + proposed-2 (cifar10)
```
python main.py --training-method train_fixmatch_withABC_contrastive  --epoch 500 --val-iteration 500 --seed 0 --dataset cifar10 --num_max 1000 --label_ratio 20 --imb_ratio 100 --gpu 0 --imbalancetype long --low-dim 64 --temperature 0.07 --DA False --using-mask True --selfcon-pos False,False --contrastive-supervised True --lr 0.002 --singleView False1 --train-strong True --lambda-Scontrast .1 --mu 1 --contrastive-unlabeled supervised --contrastive-labeled supervised --eval ema
```

### fixmatch + ABC + proposed-2 (cifar100)
```
python main.py --training-method train_fixmatch_withABC_contrastive  --epoch 500 --val-iteration 500 --seed 0 --dataset cifar100 --num_max 200 --label_ratio 40 --imb_ratio 20 --gpu 0 --imbalancetype long --low-dim 64 --temperature 0.07 --DA False --using-mask True --selfcon-pos False,False --contrastive-supervised True --lr 0.002 --singleView False1 --train-strong True --lambda-Scontrast .1 --mu 1 --contrastive-unlabeled supervised --contrastive-labeled supervised --eval ema
```
