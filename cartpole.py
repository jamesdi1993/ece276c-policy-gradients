from policy_network import PolicyNetwork
from torch.distributions.categorical import Categorical
from gradient_func import gfunc1, gfunc2, gfunc3, perform
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Device is: %s" % device)
    env = gym.make("CartPole-v1")
    steps = 500
    n = 200
    g = 0.99
    pi = PolicyNetwork(4, 256, 256, 2).to(device)

    optimizer = torch.optim.Adam(pi.parameters(), lr=1e-4)
    print("The action space for Cartpole is: %s" % env.action_space)
    gf = gfunc1
    returns = np.zeros(n)
    for i in range(200):
        mu = np.mean(returns[:i])
        std = np.std(returns[:i])
        if np.isnan(mu): mu = 0
        if np.isnan(std): std = 0
        loss, G = batch_roll_out(env, pi, g, steps, device, gf)
        returns[i] = G
        # how should I do gradient descent on the data;
        print("Iteration: %s; Loss: %s" % (i, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    testPolicy(env, pi, device, g)
    plot_return(returns, n)
    env.close()

def batch_roll_out(env, pi, g, n, device, gfunc, mu=None, std=None):
    """
    Roll out n steps based on the current policy and starting state;
    :param env: The environment;
    :param g: The discount factor;
    :param n: The number of steps to break;
    :param device: The device to perform optimization
    :param gfunc: the function for calculating gradient
    :return: The \textit{cost}for the n steps;
    """
    T = env._max_episode_steps # maximum number of steps
    c = 0
    observation = env.reset()
    J = 0 # total cost
    G_total = 0
    G = torch.zeros(T) # discounted rewards for each step in one episode;
    s_log = torch.zeros(T) # log likelihoods for each step in one episode;
    t = 0 # step in episodes
    l = 0 # number of episode
    done = False
    while c < n:
        observation = torch.tensor(observation, device=device).to(device).float()
        p = pi(observation)
        m = Categorical(p)
        a = m.sample()
        observation, reward, done, info = env.step(a.numpy())
        # print("Reward: %s, Log prob: %s" % (reward * (g ** t), m.log_prob(a)))
        G[t] = reward * (g ** t)
        s_log[t] = m.log_prob(a)
        c += 1
        t += 1
        if done:
            observation = env.reset()
            if mu != None:
                J += gfunc(G, s_log, mu, std)  # add one more episode if episode is not done;
            else:
                J += gfunc(G, s_log)
            G_total += torch.sum(G)
            t = 0
            l += 1
            G = torch.zeros(T)
            s_log = torch.zeros(T)
    if not done:
        if mu != None:
            J += gfunc(G, s_log, mu, std) # add one more episode if episode is not done;
        else:
            J += gfunc(G, s_log)
        G_total += torch.sum(G)
        l += 1
    print("Number of episodes: %s; Number of steps: %s" % (l, c))
    return -J / l, G_total / l

def plot_return(G, n):
    x = np.arange(0, n, 1)
    plt.plot(x, G)
    plt.title('Average return')
    plt.show()


def testPolicy(env, pi, device, g, ):
    observation = env.reset()
    maxIter = env._max_episode_steps
    total_reward = 0
    for i in range(maxIter):
        observation = torch.tensor(observation, device=device).float()
        p = pi.forward(observation)
        m = Categorical(p)
        a = m.sample()
        observation, reward, done, info = env.step(a.detach().numpy())
        total_reward += (g ** i) * reward
        time.sleep(0.1)
        env.render()
        if done:
            break
    print("Total reward is: %s" %  total_reward)


if __name__ == '__main__':
    main()