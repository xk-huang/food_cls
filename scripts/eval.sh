for file in saved/food_resnet50_stage1_mixup_alpha_1*/ckps/*.pth.tar; do
echo $file
CUDA_VISIBLE_DEVICES=0 python eval.py --cfg config/food/food_resnet50_stage1_mixup_test.yaml resume $file 
done


for file in saved/food_resnet50_stage2*/ckps/*.pth.tar; do
echo $file
CUDA_VISIBLE_DEVICES=0 python eval.py --cfg config/food/food_resnet50_stage2_mislas_test.yaml resume $file 
done