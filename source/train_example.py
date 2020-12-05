from arguments import get_args
from ppo_agent import ppo_agent
from maddpg import MADDPG

from models import cnn_net, fc_net
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gfootball.env as football_env
from gfootball.env.wrappers import FrameStack
import os

# create the environment
def create_single_football_env(args):
    """Creates gfootball environment."""
    env = football_env.create_environment(\
            env_name=args.env_name, representation='extracted', number_of_left_players_agent_controls=args.ln_agents, number_of_right_players_agent_controls=args.rn_agents, rewards='scoring', write_video=True, write_full_episode_dumps=True, logdir='./logs/videos', dump_frequency=args.dump_frequency)#, with_checkpoints=False, 
            #)
    # env = football_env.create_environment(\
    #         env_name=args.env_name, representation='extracted', rewards='scoring', write_video=True, write_full_episode_dumps=True, logdir='./logs/videos', dump_frequency=25)#, with_checkpoints=False, 
    #         #)
    env = FrameStack(env,args.history)
    
    return env

if __name__ == '__main__': 
    # get the arguments
    args = get_args()
    # create environments
    

    env = create_single_football_env(args)

    print(env.action_space)

    maddpg_trainer = MADDPG(args, env)
    maddpg_trainer.learn()

    env.close()

    # envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)], context=None)
    # mappo_trainer = mappo_agent(envs, args)
    # mappo_trainer.learn()

    # env.close()

    # envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)], context=None)
    # ppo_trainer = ppo_agent(envs, args)
    # ppo_trainer.learn()

    # # close the environments
    # envs.close()
