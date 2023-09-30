import numpy as np
np.set_printoptions(precision=3)

P_MDP = np.array([
    # H  M  L  T  0
    [.5, .5,  0,  0,  0],  # h_dock
    [.4, .4,  0, .2,  0],  # h_clean
    [.5,  0, .5,  0,  0],  # m_dock
    [ 0, .4, .4, .2,  0],  # m_clean
    [.5,  0,  0,  0, .5],  # l_dock
    [ 0,  0, .4, .2, .4],  # l_clean
    [ 0,  0,  0,  1,  0],  # success
    [ 0,  0,  0,  0,  1]   # failure
    ])

iter_max = 1000

# value iteration finding max
x = np.array([0, 0, 0, 1, 0]).transpose()
for _ in range(iter_max):
    x = np.dot(P_MDP, x)
    x = np.array([ max(x[[0,1]]), max(x[[2,3]]), max(x[[4,5]]), x[6], x[7] ]).transpose()
print("Max:")
print(x)
# one more iteration to see values of each action
print(np.dot(P_MDP, x))

# value iteration finding min
x = np.array([0, 0, 0, 0, 1]).transpose()
for _ in range(iter_max):
    x = np.dot(P_MDP, x)
    x = np.array([ min(x[[0,1]]), min(x[[2,3]]), min(x[[4,5]]), x[6], x[7] ]).transpose()
print("Min:")
print(x)
# one more iteration to see values of each action
print(np.dot(P_MDP, x))
print("1 - Min:")
print(1 - np.dot(P_MDP, x))