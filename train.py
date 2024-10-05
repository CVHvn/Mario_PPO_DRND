import argparse
import torch

from src.environment import *
from src.memory import *
from src.model import *
from src.agent import *
from src.running_mean_std import *

def get_args():
    parser = argparse.ArgumentParser(
        """PPO implement to playing Super Mario Bros""")
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--pretrained_model", type=str, default="best_model.pth", help = 'Pretrained model path')

    parser.add_argument('--num_envs', type=int, default=32, help='Number of environment')
    parser.add_argument('--learn_step', type=int, default=768, help='Number of steps between training model')
    parser.add_argument('--batch_size', type=int, default=768, help='batch_size')
    parser.add_argument('--num_epoch', type=int, default=10, help='epoch')

    parser.add_argument('--learning_rate', type=float, default=7e-5)
    parser.add_argument('--gamma_int', type=float, default=0.99, help='Discount factor for intrinsic rewards')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for extrinsic rewards')
    parser.add_argument('--V_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.05, help='Entropy loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--clip_param', type=float, default=0.2, help='Clip coefficient for policy loss')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda coefficient')
    parser.add_argument('--target_kl', type=float, default=0.05, help='Set target_kl = None if you do not use this. Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping.')

    parser.add_argument('--ext_adv_coef', type=float, default=1, help='GAE lambda coefficient')
    parser.add_argument('--int_adv_coef', type=float, default=0.5, help='GAE lambda coefficient')

    parser.add_argument('--update_proportion', type=float, default=1, help='Update proportion for DRND models')
    parser.add_argument('--alpha_DRND1', type=float, default=0.9, help='alpha1 for DRND')
    parser.add_argument('--alpha_DRND2', type=float, default=0.1/512, help='alpha2 for DRND')
    parser.add_argument('--num_target_model', type=int, default=5, help='number target model')

    parser.add_argument("--is_normalize_advantage", type=bool, default=False, help = "Use normalize advantage or not")
    parser.add_argument("--V_loss_type", type=str, default="huber", help = "Use mse or huber loss for value network")

    parser.add_argument('--total_step', type=int, default=int(5e6), help='Total step for training')
    parser.add_argument('--save_model_step', type=int, default=int(1e5), help='Number of steps between saving model')
    parser.add_argument('--save_figure_step', type=int, default=768, help='Number of steps between testing model')
    parser.add_argument('--total_step_or_episode', type=str, default='step', help='choice stop training base on total step or total episode')
    parser.add_argument('--total_episode', type=int, default=None, help='Total episodes for training')

    parser.add_argument("--action_dim", type=int, default=12, help='12 if set action_type to complex else 7')
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--state_dim", type=tuple, default=(4, 84, 84))

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--additional_bonus_state_8_4_option", type=str, default="no", help = 'Option to add more bonus reward for state 8-4, this value can set to no (Do not add more reward), right_pipe (Add +50 bonus reward when Mario go to right pipe)')
    args = parser.parse_args()
    return args


def train(config):
    envs = MultipleEnvironments(config.world, config.stage, config.action_type, config.num_envs, config.additional_bonus_state_8_4_option)
    
    model = Model(config.state_dim, config.action_dim)
    target_models = []
    for i in range(config.num_target_model):
        target_models.append(Target_Model(config.state_dim, config.action_dim))
    predict_model = Feature_Model(config.state_dim, config.action_dim)

    agent = Agent(envs = envs, world = config.world, stage = config.stage, action_type = config.action_type, num_envs = config.num_envs,
              state_dim = config.state_dim, action_dim = config.action_dim, save_dir = config.save_dir,
              save_model_step = config.save_model_step, save_figure_step = config.save_figure_step, learn_step = config.learn_step,
              total_step_or_episode = config.total_step_or_episode, total_step = config.total_step, total_episode = config.total_episode,
              model = model, target_models = target_models, predict_model = predict_model, gamma = config.gamma, gamma_int = config.gamma_int,
              learning_rate = config.learning_rate, entropy_coef = config.entropy_coef, V_coef = config.V_coef,
              max_grad_norm = config.max_grad_norm, clip_param = config.clip_param, batch_size = config.batch_size,
              num_epoch = config.num_epoch, is_normalize_advantage = config.is_normalize_advantage, V_loss_type = config.V_loss_type,
              target_kl = config.target_kl, gae_lambda = config.gae_lambda, ext_adv_coef = config.ext_adv_coef,
              int_adv_coef = config.int_adv_coef, num_target_model = config.num_target_model,
              alpha_DRND1 = config.alpha_DRND1, alpha_DRND2 = config.alpha_DRND2, update_proportion = config.update_proportion,
              additional_bonus_state_8_4_option = config.additional_bonus_state_8_4_option, 
              device = "cuda:0" if torch.cuda.is_available() else "cpu")
    agent.train()

if __name__ == "__main__":
    config = get_args()
    train(config)