python test.py  \
--dataset ACDC \
--cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
--volume_path '../data/ACDC/test' \
--list_dir "../data/ACDC/lists_ACDC" \
--volume_path '../data/ACDC/test' \
--output_dir './results/acdc' \
--test_save_dir './results/acdc/predictions' \
--max_epochs 300 \
--img_size 224 \
--is_savenii
#--dataset Synapse \
#--cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
#--max_epochs 300 \
#--img_size 224 \
#--is_savenii








# To add acdc, revise train.py, trainer.py, dataset_synapse.py, my_train.sh. and add metrics.py