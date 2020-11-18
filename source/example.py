
import gfootball.env as football_env
import numpy as np
import matplotlib.pyplot as plt
from gfootball.env import wrappers
# env = football_env.create_environment(env_name="academy_run_pass_and_shoot_with_keeper", representation='extracted', number_of_left_players_agent_controls=1, stacked=False, logdir='./tmp/football', rewards='scoring', write_goal_dumps=False, write_full_episode_dumps=False, render=True)
env = football_env.create_environment(\
            env_name="academy_run_pass_and_shoot_with_keeper", stacked=False, representation='extracted', number_of_left_players_agent_controls=1, rewards='scoring', render=True, write_video=True, write_full_episode_dumps=True, logdir='./logs')#, with_checkpoints=False, 
            #)
env = wrappers.FrameStack(env,2)
# env.render(mode='rgb_array')
prev_obs = env.reset()
steps = 0
for i in range(10):
  while True:
    obs, rew, done, info = env.step(env.action_space.sample())

    env.render()
    obs = obs
    # for i in range(obs.shape[-1]):
    #   # print(prev_obs[:,:,i].shape)
    #   plt.imshow(obs[:,:,0,i])
    #   plt.show()
    #   plt.imshow(obs[:,:,1,i])
    #   plt.show()
      # if np.equal(prev_obs[:,:,i], obs[:,:,i]).all():
      #   print(i)
    
    # print(len(obs), obs.shape)
    # for key in obs[0]:
    #     # print(key)
    #     if not np.equal(obs[0][key],obs[1][key]).all():
    #         print(key)
    print(obs.shape, rew.shape, done, info)

    prev_obs = obs
    # print(obs[0])
    # print(obs[1])

    # for i in range(obs.shape[1]):
    #     if obs[0][i] != obs[1][i]:
    #         print(i,obs[0][i],obs[1][i])

    
    steps += 1
    if steps % 100 == 0:
      exit(0)
      print("Step %d Reward: %f" % (steps, rew))
    if done:
      break
  env.reset()
  # print(1)
print("Steps: %d Reward: %.2f" % (steps, rew))