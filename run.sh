#!/bin/sh


##echo "1"
##python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --gpu $5 --shifting 1
##echo "1.1"
##python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --gpu $5 --shifting 1.1
##echo "1.2"
##python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --gpu $5 --shifting 1.2
##echo "1.3"
##python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --gpu $5 --shifting 1.3
##echo "1.4"
##python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --gpu $5 --shifting 1.4
##echo "1.5"
##python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --gpu $5 --shifting 1.5


echo "0%"
python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3  
echo "2%"
python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --shifting 1 --var 50
echo "4%"                                                                             
python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --shifting 1 --var 25
echo "6%"                                                                             
python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --shifting 1 --var 16.67
echo "8%"                                                                      
python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --shifting 1 --var 12.5
echo "10%"                                                                             
python3 main.py --arch $1 --resume $2 -e 1 -q 1 -m $3 --rate $4 --shifting 1 --var 10
