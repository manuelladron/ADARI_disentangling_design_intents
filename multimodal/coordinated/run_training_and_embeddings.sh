#!/bin/bash

jupyter nbconvert --to script F20_MMML_Train_TwoWayNet_Workflow.ipynb

python3 F20_MMML_Train_TwoWayNet_Workflow.py
rm F20_MMML_Train_TwoWayNet_Workflow.py

