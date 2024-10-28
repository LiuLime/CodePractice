# For testing train program at a small set
# python3 main_singleGPU.py --mode train --approach nn --num_epochs 5 --save_every_n_epoch 1 --batch_size 512 --run_small_set

# For training on full dataset
# nohup python3 main_singleGPU.py --mode train --approach nn --num_epochs 50 --save_every_n_epoch 5 --batch_size 512 --save_dataset_idx > "20241017_nn_train.log" 2>&1 &
# nohup python3 main_singleGPU.py --mode train --approach nn --num_epochs 50 --save_every_n_epoch 1 --batch_size 512 --save_dataset_idx > "20241018_nn_train.log" 2>&1 &
# nohup python3 main_singleGPU.py --mode train --approach nn --num_epochs 50 --save_every_n_epoch 1 --batch_size 512 --save_dataset_idx --early_stopping > "20241021_nn_train.log" 2>&1 &
nohup python3 main_singleGPU.py --mode train --approach nn --architecture twoCNN_with_dropout --num_epochs 50 --save_every_n_epoch 1 --patience 10 --batch_size 32 --save_dataset_idx --early_stopping --scaling > "20241028_nn_v6_train.log" 2>&1 &

# For training on full dataset from saved epoch point
# python3 main_singleGPU.py --mode train --approach nn --num_epochs 50 --save_every_n_epoch 5 --batch_size 512 --run_from_ckp

# For testing on test set
# nohup python3 main_singleGPU.py --mode test --approach nn --architecture fiveCNN_with_dropout --batch_size 512 --test --scaling > "20241024_nn_v4_test.log" 2>&1 &


# For finding learning rate
# nohup python3 main_singleGPU.py --mode lrfinder --approach nn --architecture twoCNN_with_dropout --batch_size 512 --scaling > "20241025_nn_v6_lrfinder.log" 2>&1 &