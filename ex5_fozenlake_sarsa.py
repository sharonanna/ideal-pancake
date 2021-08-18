from mpl_toolkits.mplot3d.axes3d import Axes3D
import gym
from gym import wrappers
import numpy as np
import random
from matplotlib import pyplot as plt

def V(s):
    return (1-epsilon) * np.max(Q[s,:]) + epsilon * np.mean(Q[s,:])

"""
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""
# Deterministic environment

from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv(is_slippery=False)
#env = FrozenLakeEnv(map_name="8x8")
#env = gym.make("FrozenLake-v0")

num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros([num_states, num_actions])
#Q = np.random.randn(num_states,num_actions)

num_episodes = 100
rewardvector = []
gamma = 0.5
alpha = 0.8
epsilon = 0.5

#print(Q.shape)
def epsilon_policy(state,Q,epsilon):
    a = np.argmax(Q[state, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(num_actions)
        #a = env.action_space.sample()
    return a

averageepisodelength = []
for i in range(num_episodes):
    episodelength = 0
    state = env.reset()
    totalreward=0

    rand = np.random.randn(1, env.action_space.n)
    #action = random.randint(0,num_actions-1)
    done = False
    action = epsilon_policy(state, Q,epsilon)
    #print(state,action)
    while not done:
        newstate, reward, done , q = env.step(action)
        #print(newstate, reward, done , q)
        newaction = epsilon_policy(newstate,Q,epsilon)
        #newaction = np.argmax(Q[newstate, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)) )
        #print("A:",newaction)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[newstate, newaction] - Q[state, action])
        totalreward += reward

        state = newstate
        action = newaction
        episodelength += 1
    rewardvector.append(totalreward)
    averageepisodelength.append(episodelength)
    if i % 500 ==0 and i is not 0:
        print("Average episode length",np.mean(averageepisodelength))
       # print("Success rate: ", (sum(rewardvector) / i))
        averageepisodelength = []


#for state in range(num_states):
#    statevalue[state] = np.max(Q[state,:])

statevalue = np.asarray([V(s) for s in range(num_states)])

print("Reward for each episode:",rewardvector)
print("Q function: \n",Q)


optimal_policy = np.asarray([np.argmax(Q[i,:]) for i in range(num_states)])
print(optimal_policy.reshape(int(num_states/num_actions),num_actions))

#print("Optimal policy: \n",optimal_policy)
#print("Success rate: " ,(sum(rewardvector) / num_episodes))

def printpolicy(pi):
    arrows = ["\t←\t", "\t↓\t", "\t→\t", "\t↑\t"]
    size = int(np.sqrt(len(pi)))
    for i in range(size):
        row = "|"
        for j in range(size):
            row += arrows[pi[i*size+j]] + "|"
        print(row)

printpolicy(optimal_policy)

fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(111, projection='3d')
x = [i for i in range(num_states)]
y = [i for i in range(num_actions)]
XP, YP = np.meshgrid(x, y)
zs = np.array([ Q[x,y] for x,y in zip(np.ravel(XP), np.ravel(YP))])
Z = zs.reshape(XP.shape)
p = ax.plot_surface(XP, YP, Z, rstride=4, cstride=4, linewidth=0)
plt.xlabel("States")
plt.ylabel("Actions")
plt.show()


plt.plot([i for i in range(num_states)], statevalue, 'g*-')
plt.xlabel("States")
plt.show()

env.close()

