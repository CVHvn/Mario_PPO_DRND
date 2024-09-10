# Mario_PPO_RND
Playing Super Mario Bros with Proximal Policy Optimization (PPO) and Distributional Random Network Distillation (DRND)

## Introduction

My PyTorch Proximal Policy Optimization (PPO) + Distributional Random Network Distillation (DRND) implement to playing Super Mario Bros. There are [PPO paper](https://arxiv.org/abs/1707.06347) and [DRND paper](https://arxiv.org/abs/2401.09750).
<p align="center">
  <img src="demo/gif/8-4-10-targets.gif" width="200"><br/>
  <i>World 8-4 trained with 10 target networks without more reward for correct pipe</i>
</p>

<p align="center">
  <img src="demo/gif/8-4-correct-pipe-reward-5-targets.gif" width="200"><br/>
  <i>World 8-4 trained with 5 target networks and more reward when Mario goes down with correct pipe</i>
</p>

<p align="center">
  <img src="demo/gif/8-4-correct-pipe-reward-10-targets.gif" width="200"><br/>
  <i>World 8-4 trained with 10 target networks and more reward when Mario goes down with correct pipe</i>
</p>

## Motivation

I completed all worlds of Super Mario Bros with (PPO + RND) [https://github.com/CVHvn/Mario_PPO_RND], but Random Network Distillation has a problem: it depends on random weights for the target and prediction networks in the RND system. Recently, I read the DRND paper and saw that DRND solves this problem because it uses multiple target networks, which eliminates the concern about poor initialization. Additionally, I noticed that DRND yields better results in benchmark reports, and I want to try this with Mario.

## How to use it

You can use my notebook for training and testing agent very easy:
* **Train your model** by running all cell before session test
* **Test your trained model** by running all cell except agent.train(), just pass your model path to agent.load_model(model_path)

Or you can use **train.py** and **test.py** if you don't want to use notebook:
* **Train your model** by running **train.py**: For example training for stage 1-4: python train.py --world 1 --stage 4 --num_envs 8
* **Test your trained model** by running **test.py**: For example testing for stage 1-4: python test.py --world 1 --stage 4 --trained_model best_model.pth --num_envs 2

## Trained models

You can find trained model in folder [trained_model](trained_model)

## Hyperparameters

Because normal PPO can complete 31/32 stages with faster training times (when I use RND), I don't want to waste time. I will only try this algorithm with stage 8-4.

My hyperparameter experience:
- I tried entropy coefficient values of 0.1 and 0.5 and found that 0.5 performs better. With 0.1, Mario might not be able to complete world 8-4.
- I experimented with 5 and 10 target networks, and found that 5 networks help the model converge better. However, 10 networks might help the model complete this stage when I remove the additional reward for Mario going to the correct pipe. (But more training might be needed to have conclusions, as with 10 target networks, the win rate for stage 8-4 is still not 100% without the correct pipe additional reward).
- I tried intrinsic advantage coefficients of 0.5 and 1, and found that the model cannot complete the stage (or at least requires a lot of steps).
- I did not change other hyperparameters compared to PPO + RND.
- I set update_proportion to 1 because we do not face the problem of the prediction network converging too quickly.

## Requirements

* **python 3>3.6**
* **gym==0.25.2**
* **gym-super-mario-bros==7.4.0**
* **imageio**
* **imageio-ffmpeg**
* **cv2**
* **pytorch** 
* **numpy**

## Acknowledgements
With my code, I can completed all 32/32 stages of Super Mario Bros. This code included new custom reward system (for stage 8-4) and PPO+DRND for agent training.

## Reference
* [yk7333 DRND](https://github.com/yk7333/DRND)
* [CVHvn PPO+RND](https://github.com/CVHvn/Mario_PPO_RND)
* [Stable-baseline3 PPO](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO)
* [lazyprogrammer A2C](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl3/a2c)
* [jcwleo RND](https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py)
* [DI-engine RND](https://opendilab.github.io/DI-engine/12_policies/rnd.html)