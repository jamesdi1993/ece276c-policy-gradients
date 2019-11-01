import gym
import numpy as np
import time

env = gym.make("CartPole-v1")
n = 1000

env.reset()
print("The action space for Cartpole is: %s" % env.action_space)
print("The observation space for Cartpole is: (%s, %s)" % (env.observation_space[0], env.observation_space[1]))
for i in range(n):
    observation, reward, done, info = env.step(env.action_space.sample())
    time.sleep(0.05)
    env.render()
    if done:
        break
env.close()

