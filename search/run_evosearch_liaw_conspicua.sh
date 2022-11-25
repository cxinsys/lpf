nohup python evosearch_liaw.py --config ../config/evosearch/config_search_conspicua_gpu.yaml --gpu 2 > log_search_evo_gpu2.out 2>&1 &
sleep 5s

nohup python evosearch_liaw.py --config ../config/evosearch/config_search_conspicua_gpu.yaml --gpu 3 > log_search_evo_gpu3.out 2>&1 &
