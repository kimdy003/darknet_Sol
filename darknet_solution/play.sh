#!/bin/bash

./densenet_20_serial

sleep 3

./vgg_15_serial

sleep 3

./alex_20_serial

sleep 3

./multi_net_serial



#for((i=0; i<4; i++)); do
#done

