python evaluate_multi.py \
--model-name resnet50dsbw \
--exp-setting office \
--model-path /root/autodl-tmp/output/office_resnet50dsbw_GW_all_5_64_amazon_webcam_SGD_0.01_yes_40/best_resnet50dsbw+None+i0_amazon2webcam.pth \
--args-path /root/autodl-tmp/output/office_resnet50dsbw_GW_all_5_64_amazon_webcam_SGD_0.01_yes_40/args_dict.pth \
--num-classes 31 \
--source-datasets amazon \
--target-dataset webcam \



# --model-name resnet50dsbw \  #模型名称[resnet50dsbn,resnet50dsbw]
# --exp-setting office \       # office visda
# --model-path /root/autodl-tmp/output/office_resnet50dsbw_GW_all_5_64_amazon_webcam_SGD_0.01_yes_40/best_resnet50dsbw+None+i0_amazon2webcam.pth \ #模型参数保存文件路径
# --args-path /root/autodl-tmp/output/office_resnet50dsbw_GW_all_5_64_amazon_webcam_SGD_0.01_yes_40/args_dict.pth \  #实验参数文件路径
# --num-classes 31 \  #类别个数
# --source-datasets amazon \  # [amazon,webcam,dslr]
# --target-dataset webcam \   # [amazon,webcam,dslr]

