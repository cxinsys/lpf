nohup python search_isl_haxyridis.py --config config_search_succinea.yaml --gpu 0 > log_search_gpu0.out 2>&1 &
sleep 1s

nohup python search_isl_haxyridis.py --config config_search_succinea.yaml --gpu 1 > log_search_gpu1.out 2>&1 &
sleep 1s

nohup python search_isl_haxyridis.py --config config_search_succinea.yaml --gpu 2 > log_search_gpu2.out 2>&1 &
sleep 1s

nohup python search_isl_haxyridis.py --config config_search_succinea.yaml --gpu 3 > log_search_gpu3.out 2>&1 &
sleep 1s

