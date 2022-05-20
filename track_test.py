from TrackEnv import TrackEnv
from policy import DoubleDQN
from utils import *

def play_qlearning(env, policy, train=False, render=False):
    episode_reward = np.zeros(env.agent_num)
    observations = env.reset()
    i = 0
    while True:
        if render:
            env.render(i)
        actions = [policy.decide(observation) for observation in observations]
        next_observations, rewards, dones, flag_track = env.step(actions)
        episode_reward += rewards
        if flag_track:
            print('Success!')
            break
        if dones.all():
            break
        observations = next_observations
        i += 1
    return episode_reward

if __name__ == '__main__':
    env = TrackEnv(size = 15, block_num = 5, agent_num = 3, block_size = 2)
    net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.01}
    policy = DoubleDQN(env, net_kwargs, gamma=0.99)

    policy.epsilon = 0
    policy.load()

    for i in range(10):
        print('--------------------------------')
        play_qlearning(env, policy)
        env.render(i)
    print(env.arrive, env.collision, env.overtime)