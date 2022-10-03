# Non-local Patch Mixup for Unsupervised Domain Adaptation


This is a Pytorch implementation of "Non-local Patch Mixup for Unsupervised Domain
Adaptation".


## Install

`pip install -r requirements.txt`


## Training
(1) To run the first stage and use office-31 dataset:

`python StageOne.py --gpu_id 0 --dset office --mix 0 --s 2 --max_epoch 20
 `

(2) To run the last two stages and use office-31 dataset:

`python StageTwoAndThree.py --gpu_id 0 --dset office --s 2 --t 0 --max_epoch 30 --K 4 
 `
