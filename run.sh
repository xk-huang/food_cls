CUDA_VISIBLE_DEVICES=3 python train_stage1.py --cfg config/food/food_resnet50_stage1_mixup.yaml
CUDA_VISIBLE_DEVICES=3 python train_stage2.py --cfg config/food/food_resnet50_stage2_mislas.yaml resume saved/food_resnet50_stage1_mixup_202111231816/ckps/model_best.pth.tar