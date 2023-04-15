export CUBLAS_WORKSPACE_CONFIG=:16:8

python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 100 \
--lr 0.05 --wd 0.0005 \
--lr_scheduler \
--seed 0 \
--fig_name lr=0.05-e=100-bc.png \
--save_images \
--test \
--plot
