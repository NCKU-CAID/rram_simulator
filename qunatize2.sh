python main.py --resume saved_models/resnet34_stop_gradient_10%_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 10 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_20%_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 20 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_30%_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 30 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_40%_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 40 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_50%_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 50 --arch resnet34 --epochs 20 --gpu 1

python main.py --resume saved_models/resnet34_stop_gradient_10%_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 10 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_20%_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 20 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_30%_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 30 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_40%_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 40 --arch resnet34 --epochs 20 --gpu 1
python main.py --resume saved_models/resnet34_stop_gradient_50%_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 50 --arch resnet34 --epochs 20 --gpu 1
