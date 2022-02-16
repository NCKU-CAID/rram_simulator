## train a network
python main.py --arch vgg8
python main.py --arch vgg11
python main.py --arch alexnet
python main.py --arch resnet34

## quntize to 4,5,6,7,8 bit and quantize different model
python main.py --resume saved_models/vgg8_original.py --finetune 1 -q 1 -m linear -w 4 --arch vgg8 
python main.py --resume saved_models/vgg8_original.py --finetune 1 -q 1 -m linear -w 5 --arch vgg8 
python main.py --resume saved_models/vgg8_original.py --finetune 1 -q 1 -m linear -w 6 --arch vgg8 
python main.py --resume saved_models/vgg8_original.py --finetune 1 -q 1 -m linear -w 7 --arch vgg8 
python main.py --resume saved_models/vgg8_original.py --finetune 1 -q 1 -m linear -w 8 --arch vgg8 
## change name into vgg8_8bit.pt, vgg8_7bit.pt .......



## prune different network and different bitwidth
python main.py --resume saved_models/vgg8_4bit.py -e 1 --placement 1 --prune 1 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_5bit.py -e 1 --placement 1 --prune 1 --prune_ratio 20 --arch vgg8
python main.py --resume saved_models/vgg8_6bit.py -e 1 --placement 1 --prune 1 --prune_ratio 30 --arch vgg8
python main.py --resume saved_models/vgg8_7bit.py -e 1 --placement 1 --prune 1 --prune_ratio 40 --arch vgg8
python main.py --resume saved_models/vgg8_8bit.py -e 1 --placement 1 --prune 1 --prune_ratio 50 --arch vgg8


## retrain the pruned_network 
python main.py --reumse saved_models/vgg8_pruned10%.pt --finetune 1 --stop_gradient 1 --prune_ratio 10 --arch vgg8 
python main.py --resume saved_models/vgg8_pruned20%.pt --finetune 1 --stop_gradient 1 --prune_ratio 20 --arch vgg8
python main.py --resume saved_models/vgg8_pruned30%.pt --finetune 1 --stop_gradient 1 --prune_ratio 30 --arch vgg8
python main.py --resume saved_models/vgg8_pruned40%.pt --finetune 1 --stop_gradient 1 --prune_ratio 40 --arch vgg8
python main.py --resume saved_models/vgg8_pruned50%.pt --finetune 1 --stop_gradient 1 --prune_ratio 50 --arch vgg8


## quantize the retrain_network to  4,5,6,7,8 bit
python main.py --resume saved_models/vgg8_stop_gradient_10linear_ideal.pt --finetune 1 -q 1 -m linear -w 4 --stop_gradient 1 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_20linear_ideal.pt --finetune 1 -q 1 -m linear -w 4 --stop_gradient 1 --prune_ratio 20 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_30linear_ideal.pt --finetune 1 -q 1 -m linear -w 4 --stop_gradient 1 --prune_ratio 30 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_40linear_ideal.pt --finetune 1 -q 1 -m linear -w 4 --stop_gradient 1 --prune_ratio 40 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_50linear_ideal.pt --finetune 1 -q 1 -m linear -w 4 --stop_gradient 1 --prune_ratio 50 --arch vgg8

python main.py --resume saved_models/vgg8_stop_gradient_10linear_ideal.pt --finetune 1 -q 1 -m linear -w 5 --stop_gradient 1 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_20linear_ideal.pt --finetune 1 -q 1 -m linear -w 5 --stop_gradient 1 --prune_ratio 20 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_30linear_ideal.pt --finetune 1 -q 1 -m linear -w 5 --stop_gradient 1 --prune_ratio 30 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_40linear_ideal.pt --finetune 1 -q 1 -m linear -w 5 --stop_gradient 1 --prune_ratio 40 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_50linear_ideal.pt --finetune 1 -q 1 -m linear -w 5 --stop_gradient 1 --prune_ratio 50 --arch vgg8

python main.py --resume saved_models/vgg8_stop_gradient_10linear_ideal.pt --finetune 1 -q 1 -m linear -w 6 --stop_gradient 1 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_20linear_ideal.pt --finetune 1 -q 1 -m linear -w 6 --stop_gradient 1 --prune_ratio 20 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_30linear_ideal.pt --finetune 1 -q 1 -m linear -w 6 --stop_gradient 1 --prune_ratio 30 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_40linear_ideal.pt --finetune 1 -q 1 -m linear -w 6 --stop_gradient 1 --prune_ratio 40 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_50linear_ideal.pt --finetune 1 -q 1 -m linear -w 6 --stop_gradient 1 --prune_ratio 50 --arch vgg8

python main.py --resume saved_models/vgg8_stop_gradient_10linear_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_20linear_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 20 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_30linear_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 30 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_40linear_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 40 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_50linear_ideal.pt --finetune 1 -q 1 -m linear -w 7 --stop_gradient 1 --prune_ratio 50 --arch vgg8

python main.py --resume saved_models/vgg8_stop_gradient_10linear_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_20linear_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 20 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_30linear_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 30 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_40linear_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 40 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_50linear_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 50 --arch vgg8



### test accuracy before split under thermal effect
python main.py --resume saved_models/vgg8_stop_gradient_10linear4%.pt -e 1 --placement 1 --tile_pairing 1 --testbit 4 --arch vgg8 
python main.py --resume saved_models/vgg8_stop_gradient_10linear5%.pt -e 1 --placement 1 --tile_pairing 1 --testbit 5 --arch vgg8 
python main.py --resume saved_models/vgg8_stop_gradient_10linear6%.pt -e 1 --placement 1 --tile_pairing 1 --testbit 6 --arch vgg8 
python main.py --resume saved_models/vgg8_stop_gradient_10linear7%.pt -e 1 --placement 1 --tile_pairing 1 --testbit 7 --arch vgg8 
python main.py --resume saved_models/vgg8_stop_gradient_10linear8%.pt -e 1 --placement 1 --tile_pairing 1 --testbit 8 --arch vgg8 


### split after prune and retrain under thermal impact
python main.py --resume saved_models/vgg8_stop_gradient_10linear4%.pt -e 1 --placement 1 --split 1 --experiment 1 --testbit 4 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_10linear5%.pt -e 1 --placement 1 --split 1 --experiment 1 --testbit 5 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_10linear6%.pt -e 1 --placement 1 --split 1 --experiment 1 --testbit 6 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_10linear7%.pt -e 1 --placement 1 --split 1 --experiment 1 --testbit 7 --prune_ratio 10 --arch vgg8
python main.py --resume saved_models/vgg8_stop_gradient_10linear8%.pt -e 1 --placement 1 --split 1 --experiment 1 --testbit 8 --prune_ratio 10 --arch vgg8


### remapping 
python main.py --resume saved_models/vgg8_8bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 8 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_7bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 7 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_6bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 6 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_5bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 5 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_4bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 4 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner


### only dwongrading
python main.py --resume saved_models/vgg8_8bit.pt -e 1 --placement 1 --downgrade 1 --testbit 8 --arch vgg8 --experiment 1


### downgrading + tile pairing (要去tile_pairing code裡面調整)
python main.py --resume saved_models/vgg8_8bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 8 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_7bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 7 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_6bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 6 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_5bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 5 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner
python main.py --resume saved_models/vgg8_4bit.pt -e 1 --placement 1 --tile_pairing 1 --testbit 4 --arch vgg8 ##要去placement function中調整擺放方式from 4 corner



 
