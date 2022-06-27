#!bin/sh

cd ./jrender
python nerf_modded.py --config ./configs/Easyship.txt

python nerf_modded.py --config ./configs/Easyship.txt --ft_path ./logs/Easyship/049999.tar --render_only
# results in ./jrender/logs/Easyship/renderonly_path_000000

# TODO conbined with jRealGAN
