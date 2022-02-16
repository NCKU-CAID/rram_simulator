




### remapping from bottom left
python main.py --resume saved_models/vgg11_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg11 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/vgg11_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg11 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/vgg11_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg11 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/vgg11_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg11 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/vgg11_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg11 --experiment 1 --direct bl --remapping 1

#python main.py --resume saved_models/vgg8_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg8 --experiment 1 --direct bl --remapping 1
#python main.py --resume saved_models/vgg8_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg8 --experiment 1 --direct bl --remapping 1
#python main.py --resume saved_models/vgg8_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg8 --experiment 1 --direct bl --remapping 1
#python main.py --resume saved_models/vgg8_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg8 --experiment 1 --direct bl --remapping 1
#python main.py --resume saved_models/vgg8_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg8 --experiment 1 --direct bl --remapping 1



python main.py --resume saved_models/alexnet_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch alexnet --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/alexnet_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch alexnet --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/alexnet_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch alexnet --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/alexnet_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch alexnet --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/alexnet_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch alexnet --experiment 1 --direct bl --remapping 1

python main.py --resume saved_models/resnet34_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch resnet34 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/resnet34_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch resnet34 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/resnet34_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch resnet34 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/resnet34_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch resnet34 --experiment 1 --direct bl --remapping 1
python main.py --resume saved_models/resnet34_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch resnet34 --experiment 1 --direct bl --remapping 1


## remapping from bottom right 
python main.py --resume saved_models/vgg11_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg11 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/vgg11_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg11 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/vgg11_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg11 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/vgg11_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg11 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/vgg11_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg11 --experiment 1 --direct br --remapping 1

#python main.py --resume saved_models/vgg8_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg8 --experiment 1 --direct br --remapping 1
#python main.py --resume saved_models/vgg8_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg8 --experiment 1 --direct br --remapping 1
#python main.py --resume saved_models/vgg8_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg8 --experiment 1 --direct br --remapping 1
#python main.py --resume saved_models/vgg8_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg8 --experiment 1 --direct br --remapping 1
#python main.py --resume saved_models/vgg8_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg8 --experiment 1 --direct br --remapping 1


python main.py --resume saved_models/alexnet_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch alexnet --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/alexnet_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch alexnet --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/alexnet_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch alexnet --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/alexnet_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch alexnet --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/alexnet_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch alexnet --experiment 1 --direct br --remapping 1

python main.py --resume saved_models/resnet34_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch resnet34 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/resnet34_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch resnet34 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/resnet34_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch resnet34 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/resnet34_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch resnet34 --experiment 1 --direct br --remapping 1
python main.py --resume saved_models/resnet34_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch resnet34 --experiment 1 --direct br --remapping 1


## remapping from top left 
python main.py --resume saved_models/vgg11_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg11 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/vgg11_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg11 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/vgg11_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg11 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/vgg11_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg11 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/vgg11_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg11 --experiment 1 --direct tl --remapping 1

#python main.py --resume saved_models/vgg8_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg8 --experiment 1 --direct tl --remapping 1
#python main.py --resume saved_models/vgg8_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg8 --experiment 1 --direct tl --remapping 1
#python main.py --resume saved_models/vgg8_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg8 --experiment 1 --direct tl --remapping 1
#python main.py --resume saved_models/vgg8_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg8 --experiment 1 --direct tl --remapping 1
#python main.py --resume saved_models/vgg8_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg8 --experiment 1 --direct tl --remapping 1


python main.py --resume saved_models/alexnet_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch alexnet --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/alexnet_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch alexnet --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/alexnet_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch alexnet --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/alexnet_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch alexnet --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/alexnet_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch alexnet --experiment 1 --direct tl --remapping 1

python main.py --resume saved_models/resnet34_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch resnet34 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/resnet34_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch resnet34 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/resnet34_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch resnet34 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/resnet34_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch resnet34 --experiment 1 --direct tl --remapping 1
python main.py --resume saved_models/resnet34_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch resnet34 --experiment 1 --direct tl --remapping 1

## remapping from top right 
python main.py --resume saved_models/vgg11_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg11 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/vgg11_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg11 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/vgg11_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg11 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/vgg11_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg11 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/vgg11_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg11 --experiment 1 --direct tr --remapping 1

#python main.py --resume saved_models/vgg8_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch vgg8 --experiment 1 --direct tr --remapping 1
#python main.py --resume saved_models/vgg8_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch vgg8 --experiment 1 --direct tr --remapping 1
#python main.py --resume saved_models/vgg8_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch vgg8 --experiment 1 --direct tr --remapping 1
#python main.py --resume saved_models/vgg8_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch vgg8 --experiment 1 --direct tr --remapping 1
#python main.py --resume saved_models/vgg8_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch vgg8 --experiment 1 --direct tr --remapping 1


python main.py --resume saved_models/alexnet_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch alexnet --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/alexnet_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch alexnet --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/alexnet_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch alexnet --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/alexnet_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch alexnet --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/alexnet_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch alexnet --experiment 1 --direct tr --remapping 1

python main.py --resume saved_models/resnet34_4bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 4 --arch resnet34 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/resnet34_5bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 5 --arch resnet34 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/resnet34_6bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 6 --arch resnet34 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/resnet34_7bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 7 --arch resnet34 --experiment 1 --direct tr --remapping 1
python main.py --resume saved_models/resnet34_8bit.pt --gpu 1 -e 1 --placement 1 --tile_pairing 0 --testbit 8 --arch resnet34 --experiment 1 --direct tr --remapping 1


