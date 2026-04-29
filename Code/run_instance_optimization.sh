CUDA_VISIBLE_DEVICES=1 python BRATS_instance_optimization.py --datapath /mnt/Drive2/lucas/models/DIRAC/Dataset/predict_gbm/ --test_run > output_inst_opt.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python BRATS_instance_optimization_alt.py --datapath /mnt/Drive2/lucas/models/DIRAC/Dataset/predict_gbm/ --test_run > output_inst_opt_alt.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=5 python BRATS_instance_optimization.py --datapath /mnt/Drive2/lucas/models/DIRAC/Dataset/predict_gbm/ > output_inst_opt.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python BRATS_instance_optimization_alt.py --datapath /mnt/Drive2/lucas/models/DIRAC/Dataset/predict_gbm/ > output_inst_opt_alt.txt 2>&1 &
