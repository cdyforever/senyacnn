#!/bin/sh
gcc -Wall -O3 bmpfile.c parser.c layer_conv.c layer_pool.c layer_misc.c layer_yolo.c nms.c net.c main.c -o test.exe -lm
