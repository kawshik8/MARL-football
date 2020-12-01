import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.993, help='the discount factor of RL')
    parse.add_argument('--history', type=int, default=4, help='history of frames to use')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--num-workers', type=int, default=8, help='the number of workers to collect samples')
    parse.add_argument('--env-name', type=str, default='academy_empty_goal_close', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=8, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate of the algorithm')
    parse.add_argument('--epoch', type=int, default=4, help='the epoch during training')
    parse.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    parse.add_argument('--vloss-coef', type=float, default=0.5, help='the coefficient of value loss')
    parse.add_argument('--ent-coef', type=float, default=0.01, help='the entropy loss coefficient')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--polyak-tau', type=float, default=0.005, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--total-frames', type=int, default=int(2e6), help='the total frames for training')
    parse.add_argument('--n-episodes', type=int, default=int(1e6), help='the total no of episodes for training')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--clip', type=float, default=0.27, help='the ratio clip param')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--lr-decay', action='store_true', help='if using the learning rate decay during decay')
    parse.add_argument('--max-grad-norm', type=float, default=5.0, help='grad norm')
    parse.add_argument('--display-interval', type=int, default=100, help='the interval that display log information')
    parse.add_argument('--log-dir', type=str, default='logs/')
    parse.add_argument('--soft-update', action='store_true', help='soft update vs hard update')

    parse.add_argument('--test-model', action='store_true', help='testing phase')

    ############ multi agent stuff ############

    parse.add_argument('--n-agents', type=int, default=2, help='no of agents to control')
    parse.add_argument('--action-dim', type=int, default=32, help='hidden dim of action rep')
    parse.add_argument('--tie-actor-wts', type=int, default=1, help='share weights for actor networks')
    parse.add_argument('--tie-critic-wts', type=int, default=1, help='share weights for critic networks')

    ############ off policy stuff ############

    parse.add_argument('--buffer-size', type=int, default=1e6, help='total number of frames in memory')
    parse.add_argument('--alpha', type=float, default=0.2, help='param for alpha value in SAC type algos')
    parse.add_argument('--reward-scale', type=float, default=10.0, help='param for reward scale')
    parse.add_argument('--start_steps', type=int, default=10000, help='param for start steps from training starts')

    args = parse.parse_args()

    return args
