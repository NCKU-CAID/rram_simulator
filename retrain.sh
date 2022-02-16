python main.py --resume saved_models/resnet34_prune10%0bit.pt --finetune 1 --stop_gradient 1 --prune_ratio 10 --arch resnet34 --epochs 100
python main.py --resume saved_models/resnet34_prune20%0bit.pt --finetune 1 --stop_gradient 1 --prune_ratio 20 --arch resnet34 --epochs 100
python main.py --resume saved_models/resnet34_prune30%0bit.pt --finetune 1 --stop_gradient 1 --prune_ratio 30 --arch resnet34 --epochs 100
