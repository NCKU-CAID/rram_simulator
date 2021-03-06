# RRAM simulator

## Experiment setup

## Check python version
```
$ python --version
Python 3.7.3
```

### Download source code
* Since `hotspot` is different project, use `recursive` to clone
```
git clone --recurse-submodules https://github.com/NCKU-CAID/rram_simulator.git
```
or 
```
git clone https://github.com/NCKU-CAID/rram_simulator.git
cd rram_simulator
git submodule update
```

### Use virtual enviroinment
* install pacakge *virtualenv* fisrt
```python=
python3 -m venv venv
source venv/bin/activate
```

### Install requirement package
```python=
pip install -r package_list
```

### Show helping list
```python=
python main.py --help
```
* `--help` : show help list

### Select training model
* VGG8, VGG11, AlexNet, ResNet34
* Pretrained model will be saved under directory `saved_models`
```python=
python main.py --arch vgg11
```
* `--arch` model architecture

## Evaluate 
```python=
python main.py --arch vgg11 --resume saved_models/vgg11.pt -e 1 
```
* `-e` : evaluate, default is `0`

### Quantize model
```python=
python main.py --resume saved_models/vgg11_ideal.pt --finetune 1 -q 1 -m linear -w 8 --arch vgg11
```
* `--resume` : pretrained model
* `--finetune`, `-q` should be **`1`** if need quantization
* `-m` : rram resistance is linear mode
* `-w` : quantization bits

### Remapping weight from 4 corners
```python=
python main.py --resume saved_models/vgg11_8bit.pt --gpu 1 -e 1 --placement 1 --testbit 8 --arch vgg11 --experiment 1 --direct bl --remapping 1
```
* `--gpu 1` : use second GPU(default is 0)
* `--tile_pairing` : compare with control group
* `test_bits` : weight bits
* `--experiment` : `1` for 1 cell represent 1 weight(default is `1`)
* `--direct` : remapping direction
  * `tl` : from top left
  * `bl` : from bottom left
  * `tr` : from top right
  * `br` : from bottom right

* `--remapping` : `1` for remapping and test accuracy under thermal effect (default is `0`)


### Prune weight in model
```python=
python main.py --resume saved_models/vgg11_8bit.pt -e 1 --placement 1 --experiment 1 --prune 1 --prune_ratio 10 --arch vgg11
```
* `--prune_ratio` : `0`~`50`, prune ratio, prune same ratio in each layer

### Retrain pruned model
```python=
python main.py --resume saved_models/vgg11_prune10%.pt --finetune 1 --stop_gradient 1 --prune_ratio 10 --arch vgg11
```
* before running `--finetune`, optimizer should be modified because the version is too old
* modified file `venv/lib/python3.7/site-packages/torch/optim/optimizer.py` as following, then this command can be run

```
164 #p.grad.detach_()
165 p.grad.detach()
```

### Quantize retrained model
```python=
python main.py --resume saved_models/vgg11_retrain_10%_ideal.pt --finetune 1 -q 1 -m linear -w 8 --stop_gradient 1 --prune_ratio 10 --arch vgg11
```
### Do weight splitting for pruned model
```python=
python main.py --resume saved_models/vgg11_stop_gradient_10linear8%.pt -e 1 --placement 1 --split 1 --experiment 1 --testbit 8 --prune_ratio 10 --arch vgg11 --direct bl
```
* `--placement` : map weight to subarray, then get thermal of subarray
* `--split` : split weight
