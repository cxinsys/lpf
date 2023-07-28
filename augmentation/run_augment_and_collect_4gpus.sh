nohup python augment_and_collect.py --config ../config/augmentation/config_augmentation_gpu_local.yaml --device "cuda:0" > log_augment_gpu0.out 2>&1 &
sleep 1s

nohup python augment_and_collect.py --config ../config/augmentation/config_augmentation_gpu_local.yaml --device "cuda:1" > log_augment_gpu1.out 2>&1 &
sleep 1s

nohup python augment_and_collect.py --config ../config/augmentation/config_augmentation_gpu_local.yaml --device "cuda:2" > log_augment_gpu2.out 2>&1 &
sleep 1s

nohup python augment_and_collect.py --config ../config/augmentation/config_augmentation_gpu_local.yaml --device "cuda:3" > log_augment_gpu3.out 2>&1 &
sleep 1s
