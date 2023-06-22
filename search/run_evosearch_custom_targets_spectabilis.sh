nohup python evosearch_custom_targets.py --config ../config/evosearch/config_search_spectabilis_gpu.yaml --gpu 3 > log_evosearch_spectabilis_3.out 2>&1 &
sleep 5s

nohup python evosearch_custom_targets.py --config ../config/evosearch/config_search_spectabilis_gpu.yaml --gpu 3 > log_evosearch_spectabilis_4.out 2>&1 &
