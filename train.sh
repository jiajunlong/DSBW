python trainval_multi.py \
--model-name resnet50dsbw \
--exp-setting office \
--source-datasets amazon \
--target-datasets webcam \
--whitening BW \
--whitening-pos last \
--iter 5 \
--pre None \
--predir  None \
--group-size 32 \
--optimizer SGD \
--learning-rate 0.01 \
--batch-size 40 \
--base-dir /root/autodl-tmp/output/

# --model-name resnet50dsbw \ #[resnet50dsbn,resnet50dsbw]
# --exp-setting office \
# --source-datasets amazon \  #[amazon, webcam, dslr]
# --target-datasets webcam \  #[amazon, webcam,dslr]
# --whitening GW \            #[GW, BW, None]
# --whitening-pos all \      #[l0, last, all, l0+last, all+last] all是指l0 + layer1-4
# --iter 5 \                  #[5,6,7,8...]
# --pre None \                #[None,yes]
# --predir None \  #[None,model/models_CVPR20/resnet50_ItN_T5_NC64_No_sgd_lr0.1_step_wd0.0001_dr0.pth]
# --group-size 32 \           #[32,64]
# --optimizer Adam \          #[Adam, SGD]
# --learning-rate 0.0001 \    #[0.0001, 0.1]
# --batch-size 40 \
# --base-dir /root/autodl-tmp/output/ #输出日志文件及模型文件的根目录