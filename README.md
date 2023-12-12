# Online-CLIP
Welcome to the repository of Online-CLIP. This is a simple project to augment the accuracy of CLIP in a continual scenario using a tractable implementation of EWC.

## Abstract
In this paper, we explore and verify the ability of CLIP to be an effective continual learner. Furthermore, we show that the zero-shot performance of CLIP, when used in a Continual Learning (CL) setting is able to outperform all SOTA methods in most aspects. However, the goal of this paper is to use the strong generalization power of CLIP as a baseline to evaluate the novel methodology presented in this paper: Online-CLIP. Traditional CL techniques to mitigate catastrophic forgetting are intractable when working with large models due to the unmanageable amounts of data that need to be stored in memory after each experience. Thus, we use an online weighted sum approach that has $O(1)$ space complexity and can outperform the baseline Continual-CLIP on the CIFAR-100 dataset for class-incremental continual learning scenarios. Apart from reproducing the results of Continual-CLIP, \textbf{ALL} of the code was implemented from scratch. In particular, writing the algorithm for the online EWC was the most challenging because it needs to be: (1) Memory efficient, and (2) gel with the other tools in the avalanche source code. 

## Instructions
Clone this repo and create a conda environment as follows:
`conda create -n cl python=3.9`

Make sure to update the submodules from the repo:
`git submodule update --init --recursive`

Then, install the requirements:
`pip install -r requirements.txt`

Download CIFAR-100 into a directory called `data/`

You may now run the training script with the following example hyperparams:
```
tmux new -s train
python train.py --lr 1e-6 --wd 0.2 --epochs 1 --use_ewc &
tmux detach
```
Grab some popcorn and relax because it might take a while...

## Project Structure
This project has been built using a popular open-source continual learning library called avalanche (it's awesome). Here is the high level description of the various components:
- dataset.py: dataset fetching, preprocessing and loading
- models.py: all model classes (only 1 for now)
- plugins.py: custom implementation of online EWC
- strategies.py: miscellaneous code necessary to integrate some customizations to mix well with avalanche tools
- utils.py: mostly parsing utils for training args
