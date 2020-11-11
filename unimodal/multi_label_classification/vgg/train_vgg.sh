#!/bin/bash

jupyter nbconvert --to script Multiclass\ VGG.ipynb

python3 Multiclass\ VGG.py
rm Multiclass\ VGG.py
