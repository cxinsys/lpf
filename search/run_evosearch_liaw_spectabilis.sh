nohup python evosearch_liaw.py --config ../config/evosearch/config_search_spectabilis_gpu.yaml --gpu 0 > log_search_evo_gpu0.out 2>&1 &
sleep 5s

nohup python evosearch_liaw.py --config ../config/evosearch/config_search_spectabilis_gpu.yaml --gpu 1 > log_search_evo_gpu1.out 2>&1 &
