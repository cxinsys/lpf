nohup python evosearch_liaw.py --config ../config/evosearch/config_search_spectabilis_gpu.yaml --gpu 2 > log_evosearch_spectabilis_1.out 2>&1 &
sleep 5s

nohup python evosearch_liaw.py --config ../config/evosearch/config_search_spectabilis_gpu.yaml --gpu 2 > log_evosearch_spectabilis_2.out 2>&1 &
