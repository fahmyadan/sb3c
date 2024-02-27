from sb3c.sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
import gymnasium as gym 
from main_rl import parse_yaml_file, parse_arguments
import numpy as np 

if __name__ == "__main__":
    args = parse_arguments()
    env_cfg = parse_yaml_file(args.env_config)
    env_kwargs = env_cfg['env_args']

    model_eval =  RecurrentPPO.load('/home/tsl/Projects/tools/simulation/RL_baselines/av_baselines/sb3c/models/21lh4jk9/model.zip')
    

    env = gym.make("intersection-v0",render_mode="rgb_array")
    env.configure(env_kwargs)
    crashed = 0 
    rewards = []
    for ep in range(100):
        rew = 0 
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model_eval.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if info['crashed']:
                crashed +=1
            
            rew+=reward
            env.render()
        rewards.append(rew)
        print(f'reward {rew} in episode {ep}')
    print(f'collision rate {crashed} \n avg reward = {np.mean(rewards)}')




