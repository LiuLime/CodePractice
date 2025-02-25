# 修改对程序可见的CUDA
export CUDA_VISIBLE_DEVICES=0,1

python3 main.py --label_column Sentiment --world_size 2 --test 