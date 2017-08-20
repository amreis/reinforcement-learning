# This is an isolated implementation of RL Training to
# balance a CartPole on OpenAI's gym.

# Some interesting things:
# - slow cooling of the learning rate alpha
# - slow cooling of exploration factor
# - gradual increase of gamma (discount factor)
# This last one has the rationale of giving more value to immediate rewards
# while we're still exploring at the beginning of training, but then
# increases to give value to distant rewards once we know a little bit of
# our environment.

# Had some problems with gradients going to +/- \infty
# constant term + outer product + selection of lower/upper triangle =>
#   bias, linear, mixed 2nd order and pure 2nd order terms in a very
#   straightworward way.

# The transposition on the feature extractors is supposed to make interpreting
# the weight vector easier, since the first weights correspond to action 0
# and the 2nd half corresponds to action 1.

# The training uses Q-learning

import numpy as np

def feature_extractor_1(state, action, normalize=True):
    # features are
    # - bias
    # - the elements of the state vector
    # - mixed multiplications between elements of state vector
    # - pure quadratic terms
    # all of that times 2 (i.e. for each of the actions)
    # I chose to leave this the way I made it in my first implementation,
    # you will notice that I do not remove repeated features here (but I do
    # in the second feature extractor).
    s_prime = np.r_[1, state]
    mat = np.outer(s_prime, s_prime)
    features = np.outer(mat.reshape(-1), np.array([0, 1]) == action).T.reshape(-1)
    if normalize:
        norm = features / (np.linalg.norm(features))
        norm[0] = int(action == 0)
        norm[25] = int(action == 1)
        return norm
    return features

def feature_extractor_2(state, action, normalize=True):
    # features are
    # - bias
    # - the elements of the state vector
    # - all second order terms (mixed and pure)
    # - sines of every first and second order term
    # - cosines of every first and second order term
    # all of that times two (i.e. for each of the actions)
    s_prime = np.r_[1, state] # dim(state) + 1
    # dim(state) (dim(state) + 1) / 2
    quadratic = np.outer(s_prime, s_prime)[np.tril_indices(s_prime.shape[0])].reshape(-1)
    # (dim(state) (dim(state) + 1) / 2) - 1
    sines = np.sin(quadratic[1:])
    cosines = np.cos(quadratic[1:])
    # dim(state)(dim(state) + 1) - 2 + dim(state)(dim(state) + 1) / 2
    state_feats = np.r_[quadratic, sines, cosines]
    # dim(state_feats) * 2
    features = np.outer(state_feats, np.array([0, 1]) == action).T.reshape(-1)
    if normalize:
        # normalize everything but the bias.
        norm = features / (np.linalg.norm(features))
        norm[0] = int(action == 0) * 1.0
        norm[state_feats.shape[0]] = int(action == 1) * 1.0
        return norm
    else:
        return features

def get_value(w, state, action):
    return np.dot(w, feature_extractor_2(state, action, True))

def pick_action(w, state, eps=0.0):
    # actions are picked in an epsilon-greedy fashion
    if np.random.uniform() < eps:
        return np.random.choice(2)
    else:
        return np.argmax([
            get_value(w, state, 0),
            get_value(w, state, 1)
            ])

def boltzmann_pick_action(w, state, temperature):
    vals = np.array([get_value(w, state, a) for a in [0, 1]])
    softmax = np.exp(vals / temperature)
    softmax /= np.sum(softmax)
    return np.random.choice(2, p=softmax)

alpha = 0.1
gamma = 0.95

import gym
env = gym.make('CartPole-v1')

# Q-learning
def update_w(w, state, action, reward, next_state, renormalize=True):
    # I used some clipping because in some cases the weight vector
    # was starting to diverge to infinity.
    # This stopped happening once I:
    # - cooled the learning rate alpha over time
    # - cooled the epsilon factor for exploration over time
    # - increased the discount gradually over time
    cur_value = get_value(w, state, action)
    best_next_value = min(max([get_value(w, next_state, a) for a in [0, 1]]), 100)
    target = reward + gamma * best_next_value - cur_value
    feats = feature_extractor_2(state, action, normalize=True)
    if renormalize:
        w_prime = w + alpha * target * feats
        #return w_prime / (0.01 * np.linalg.norm(w_prime))
        return np.minimum(np.maximum(w_prime, -1e+5), 1e+5)
    return w + alpha * target * feats

def simulate(w, render=False):
    history = []
    state = env.reset()
    done = False
    while not done:
        action = pick_action(w, state)
        new_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        history.append((state, action, reward))
        state = new_state
    return history

# w = np.random.uniform(size=50) * 10 - 5
# uniform between [-1,+1]
w = np.random.uniform(size=86) * 2 - 1.0
old_w = np.array(w)
# for debugging purposes: we'd like to take both actions approximately
# equally in the beginning of learning for exploration purposes.
visit_counter = np.array([0, 0])

def train(iterations=10000):
    global w, alpha, old_w, gamma
    c = 1
    for t in range(iterations):
        state = env.reset()
        done = False
        while not done:
            eps = (100000 / (100000 + c))
            alpha =  (5000 / (5000 + c))
            # gamma approaches .95 as c -> \infty
            gamma = (1000 + 0.95 * c) / (5000 + c)
            action = pick_action(w, state, eps)
            # action = boltzmann_pick_action(w, state, max(1, 10000 / c))
            visit_counter[action] += 1
            new_state, reward, done, _ = env.step(action)
            w = update_w(w, state, action, reward, new_state)
            c += 1
            state = new_state
        # debug prints and evaluation of greedy version of policy
        if t % (iterations // 50) == 0:
            if np.mean(np.abs(old_w - w)) < 1e-4:
                break
            old_w = np.array(w)
            histories = [simulate(w) for _ in range(10)]
            print(np.mean([len(h) for h in histories]))
            print([ [a for (s,a,r) in h] for h in histories])
            print('alpha: {}, eps: {}, gamma: {}'.format(alpha, eps, gamma))
            print('Visit Counter: {}'.format(visit_counter))
        if iterations < 100 or t % (iterations // 10) == 0:
            print(w)
            print(np.mean([len(simulate(w)) for _ in range(1000)]))

#train(100000)
# the optimal policy found by the algorithm
#w_optimal = [13.88169688,0.13274069,2.39258238,0.33545005,1.42288757,1.55481588,0.69770982,0.42457566,-0.52982298,-0.12300826,-0.63034507,0.15595444,-2.27921823,-0.74014044,1.57800626,0.23722928,-0.77959938,-0.43150532,-0.68272597,-1.12695497,-0.50647816,0.42433605,-0.07874471,0.5950862,0.62218802,0.33151675,-0.81360683,-0.45600427,-1.35698212,1.96400084,0.31129915,1.55326063,1.27538735,-0.0497833,1.50771271,1.85979238,1.92607186,3.18739434,1.88978144,1.99655295,-0.03874381,2.9544558,-0.53894907,14.00668331,0.60841993,2.03660938,-0.44513697,1.78254349,1.61235184,0.21985313,0.69993325,-0.38263453,-1.02592762,0.43698128,0.23836086,-2.72772619,-0.25779519,1.51688786,-0.31012099,-0.82344881,1.10692983,-1.16049443,-1.04965158,0.0378038,0.68709878,0.39080172,0.75896668,0.17344899,0.49654826,-0.26233172,-0.45640022,-1.08956826,1.82121266,0.23739719,2.36029537,1.34217289,-0.03343517,1.94099928,2.75782758,1.22463989,2.13722356,2.04190994,1.04246987,0.20049319,3.07153606,-0.86470319]

w_optimal = [1.43359478e+01,-2.95733224e-01,1.34194318e+00,4.40755211e-01,1.32409890e+00,2.91546580e+00,-6.16299304e-01,-6.81110627e-01,-3.62903122e-02,6.50698910e-01,-3.21153941e-01,6.86593834e-01,-1.55815539e+00,-9.43137696e-01,2.31298877e+00,-8.23719981e-02,4.56676905e-01,-7.68340240e-01,-6.00239086e-01,-6.99249872e-01,1.93389641e-01,4.41652978e-01,-5.46534121e-01,8.54205789e-02,1.08674843e-01,-2.88584852e-01,3.74718707e-01,3.14899064e-01,-1.17683623e+00,8.27035185e-01,1.46974775e+00,2.09052965e+00,1.72491900e+00,9.38606587e-02,2.70307708e+00,1.88190236e+00,2.46613974e+00,1.52143600e+00,2.31144318e+00,2.33962488e+00,-3.17717217e-01,2.17320434e+00,-2.24711863e-01,1.38887614e+01,6.20111688e-01,1.75715629e+00,-6.18304116e-01,1.42782498e+00,2.89176410e+00,-1.74108766e-03,-9.56111570e-01,-7.79637740e-02,-9.60705361e-03,5.74921855e-01,-5.98905932e-01,-1.94671075e+00,4.95610969e-01,2.20357217e+00,-1.02727361e+00,2.10868824e-01,8.01001077e-01,-8.72197529e-01,-6.97914609e-01,-1.73006226e-01,6.56228201e-01,-4.40352891e-01,-5.66971824e-01,-4.42075868e-01,1.02497902e+00,6.42295569e-01,-9.83621485e-01,-1.06040054e+00,8.37205447e-01,1.80037055e+00,1.96801378e+00,1.92313166e+00,-5.35876047e-02,2.72301807e+00,2.70564438e+00,2.53758061e+00,2.28119510e+00,2.35343218e+00,1.10794196e+00,-5.41085717e-01,3.22879267e+00,-8.08776628e-02]

print('Found w = {}'.format(w))
print('Avg duration of simulations using optimal weight vector: ', end=' ')
print(np.mean([len(simulate(w_optimal, render=True)) for _ in range(10)]))
env.close()
