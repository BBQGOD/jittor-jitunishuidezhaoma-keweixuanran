#!bin/sh

cd ./JNeRF
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_comp_Car.py

cd ..
cp ./data/nerf_synthetic/Car ./data/nerf_synthetic/CarTestAll -r
cd ./utils
python mergeDataJson.py

cd ../JNeRF
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_comp_CarTestAll.py --valid

cd ../utils
python splitIMG.py

cd ../jRealGAN
python scripts/generate_multiscale_DF2K.py --input ../utils/ref --output datasets/Car_pair/multiscale_ref
python scripts/generate_multiscale_DF2K.py --input ../utils/src --output datasets/Car_pair/multiscale_src

python scripts/extract_subimages.py --input datasets/Car_pair/multiscale_ref --output datasets/Car_pair/multiscale_sub_ref --crop_size 400 --step 200
python scripts/extract_subimages.py --input datasets/Car_pair/multiscale_src --output datasets/Car_pair/multiscale_sub_src --crop_size 400 --step 200

cd ../utils
python scaleIMG.py

cd ../jRealGAN
python scripts/generate_meta_info_pairdata.py --input datasets/Car_pair/multiscale_sub_ref_new datasets/Car_pair/multiscale_src --meta_info datasets/meta_info/meta_info_Car_sub_pair.txt
# If it consumes too much time, try to use original pyTorch version.
python jrealesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume

cp ./experiments/finetune_RealESRGANx4plus_400k/models/net_g_50000.pth ./jrealesrgan/weights/pair_weight.pth
python jinference_realesrgan.py -n pair_weight -i ../JNeRF/logs/Car/test --outscale 1.0

# results in ./jRealGAN/results
