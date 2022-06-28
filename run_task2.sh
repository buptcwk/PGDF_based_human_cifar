CUDA_VISIBLE_DEVICES=0 nohup python train_cifar_getPrior.py --preset c10.20sym --noise_type worst > c10_worst.log&&CUDA_VISIBLE_DEVICES=0 nohup python train_cifar.py --preset c10.20sym --noise_type worst > c10_worst_stage2.log && nohup python test.py --noise_type worst --dataset cifar10 > c10_worst_detect.log &  

CUDA_VISIBLE_DEVICES=1 nohup python train_cifar_getPrior.py --preset c10.20sym --noise_type rand1 > c10_rand1.log&&CUDA_VISIBLE_DEVICES=1 nohup python train_cifar.py --preset c10.20sym --noise_type rand1 > c10_rand1_stage2.log && nohup python test.py --noise_type rand1 --dataset cifar10 > c10_rand1_detect.log &  

CUDA_VISIBLE_DEVICES=2 nohup python train_cifar_getPrior.py --preset c10.20sym --noise_type aggre > c10_aggre.log&&CUDA_VISIBLE_DEVICES=2 nohup python train_cifar.py --preset c10.20sym --noise_type aggre > c10_aggre_stage2.log && nohup python test.py --noise_type aggre --dataset cifar10 > c10_aggre_detect.log & 

CUDA_VISIBLE_DEVICES=4 nohup python train_cifar_getPrior.py --preset c100.50sym --noise_type noisy100 > c100.log&