#!/bin/bash

cd eml20_session02_training

python train.py trainer.max_epochs=1

model_path=`cat eml_model.txt`

python eval.py ckpt_path=${model_path}