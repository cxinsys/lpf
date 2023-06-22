nohup python evopop_liawmodel_local.py --gpu 0 > log_evopop_gpu0.out 2>&1 &
sleep 1s

nohup python evopop_liawmodel_local.py --gpu 1 > log_evopop_gpu1.out 2>&1 &
sleep 1s

nohup python evopop_liawmodel_local.py --gpu 2 > log_evopop_gpu2.out 2>&1 &
sleep 1s

nohup python evopop_liawmodel_local.py --gpu 3 > log_evopop_gpu3.out 2>&1 &
sleep 1s
