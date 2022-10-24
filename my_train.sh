python train.py  \
--dataset ACDC \
--cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
--list_dir "../data/ACDC/lists_ACDC" \
--root_path '../data/ACDC' \
--volume_path '../data/ACDC/test' \
--output_dir './results/acdc' \
--test_save_dir './results/acdc/predictions' \
--max_epochs 300 \
--img_size 224 \
--base_lr 5E-3 \
--base_weight 5E-4 \
--num_classes 4 \
--batch_size 24 \
--n_skip 2
#--dataset Synapse \
#--cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
#--max_epochs 300 \
#--img_size 224 \
#--base_lr 1E-3 \
#--base_weight 5E-4 \
#--batch_size 24










# To add acdc, revise train.py, trainer.py, dataset_synapse.py, my_train.sh. and add metrics.py