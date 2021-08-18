import numpy as np

def nextState(s, a):
    return max(0, min(s + a, n - 1))

def q(s, an):
    s1 = nextState(s, actions[an])
    return R[s, an] + gama * V[s1]

# Defining MDP
n = 6
actions = [-1, 1]
R = np.zeros([6, 2])
R[1, 0] = 1
R[4, 1] = 5
gama = 0.2
# gama = 0.3
# gama = 0.6

# Parameters
theta = 0.01

V = np.zeros(n)
Pi = np.random.choice(actions, n)

print("Arbitrary  Starting V: ", V)
print("Randomized Starting Pi: ", Pi)

converged = False

while not converged:
    delta = 0
    for s in range(0, n):
        v = V[s]
       # print ("v:",v)
        V[s] = max(q(s, 0), q(s, 1))
       # print("V:", V[s])
        delta = max(delta, abs(V[s] - v))
    converged = delta < theta
    print("V:", V)

print("Final V: ", V)

for s in range(0, n):
    Pi[s] = actions[0] if q(s, 0) > q(s, 1) else actions[1]

print("Optimized Pi: ", Pi)



