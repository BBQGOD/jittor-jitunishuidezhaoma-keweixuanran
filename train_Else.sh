#!bin/sh

cd ./JNeRF
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_comp_Coffee.py
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_comp_Scarf.py
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_comp_Scar.py

# results in ./JNeRF/logs/*/test