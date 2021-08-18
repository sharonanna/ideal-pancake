# https://github.com/openai/gym
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
import gym
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
env = gym.make('Blackjack-v0')

discount_factor=1

usableace=True
#initialize
def policy(player_sum):
    return 0 if player_sum>=20 else 1

returns=np.zeros(10 * 10 * 2)
#stateval = list(enumerate(returns,1))
value = defaultdict(float)
returns_sum = defaultdict(float)
count = defaultdict(float)

for i in range(10000):
    obs = env.reset()
    episode = []

    done = False
    while not done:
        player_sum, dealer_card, useable_ace = obs
        nextobs, reward, done, _ = env.step(policy(player_sum))
        episode.append((obs,policy(player_sum),reward))
        obs = nextobs

    #states = [episode[i][0][0] for i in range(len(episode))]
    states = [(x[0]) for x in episode]
    #print(states)
    #print(episode)
    for state in states:
        #print (state)
        first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
        # Sum up all rewards since the first occurance
        G = sum([x[2]  for i, x in enumerate(episode[first_occurence_idx:])])
        returns_sum[state] += G
        count[state] += 1.0
        value[state] = returns_sum[state] / count[state]
        #print(V)

    #    G = reward + env.step(0 if player_sum>=20 else 1)
    #    s.append(G)
    #print(episode)
    #print(obs,reward,done,_)
print(value)
x_range = np.arange(12, 22)
y_range = np.arange(1, 11)
X, Y = np.meshgrid(x_range, y_range)
def f(x):
    return value[(x[0],x[1],usableace)]
Z = np.apply_along_axis(f, 2, np.dstack([X, Y]))
#print(np.dstack([X, Y]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z,color='grey',rstride=1,cstride=1)
ax.set_xlabel('Player Sum')
ax.set_ylabel('Dealer Showing')
ax.set_zlabel('Value')
ax.set_title("{} ace".format("usable" if usableace else "no usable"))

plt.show()

fig = plt.figure()
Z = np.apply_along_axis(lambda _: value[(_[0], _[1], not(usableace))], 2, np.dstack([X, Y]))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z,rstride=1,cstride=1)
ax.set_xlabel('Player Sum')
ax.set_ylabel('Dealer Showing')
ax.set_zlabel('Value')
ax.set_title("{} ace".format("usable" if not usableace else "no usable"))
plt.show()