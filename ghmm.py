# Author: Negar Safinianaini

# This is the implementation of the new method in the below paper published at AIME 2019:
# "Gated Hidden Markov Models for Early Prediction of Outcome of Internet-based Cognitive Behavioral Therapy"

# This implementation is intended for sequences up to length 150 and for longer ones, one should use log probabilities
# This implementation was used for binary states in HMM and EM needs only 10 iterations (this fact is published already)
# In case of having more states, one should implement the convergence criteria properly.
# Value -1 is used to represent a missing observation or data point; here we handle missing values without imputation

import numpy as np


def forward(params, observations, label=None):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    alpha = np.zeros((N, S))

    # base case
    if observations[0] != -1:
        alpha[0, :] = pi * O[:, observations[0]]
    # handling missing
    else:
        alpha[0, :] = pi

    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                transition = A[s1, s2]
                # supervised part
                if i == N - 1 and label is not None:
                    if label == s2:
                        transition = 1
                    else:
                        transition = 0
                if observations[i] != -1:
                    alpha[i, s2] += alpha[i - 1, s1] * transition * O[s2, observations[i]]
                # handling missing
                else:
                    alpha[i, s2] += alpha[i - 1, s1] * transition

    return alpha, np.sum(alpha[N - 1, :])


def backward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    beta = np.zeros((N, S))

    # base case
    beta[N - 1, :] = 1

    # recursive case
    for i in range(N - 2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                if observations[i + 1] != -1:
                    beta[i, s1] += beta[i + 1, s2] * A[s1, s2] * O[s2, observations[i + 1]]
                # handling missings
                else:
                    beta[i, s1] += beta[i + 1, s2] * A[s1, s2]

    return beta, np.sum(pi * O[:, observations[0]] * beta[0, :])


# this is a modified version of Baum_Welch
# threshold: is intended to compare with the fractional change
# policy: contains the begin and end indexes needed to calculate the fractional change e.g [[0, 9], [-10, -1]]
# label: is the hidden state of the GHMM which needs regulation by gate mechanism
# labels for the training data are expected to be at the end of each sequence in the training data
def ghmm(training, pi, A, O, iterations, threshold, policy, label):
    pi, A, O = np.copy(pi), np.copy(A), np.copy(O)
    S = pi.shape[0]
    begin = policy[0]
    end = policy[1]

    # do several steps of EM hill climbing
    for it in range(iterations):
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)

        for observations in training:
            obs = observations[:-1]

            # compute forward-backward matrices
            alpha, za = forward((pi, A, O), obs, observations[-1]) # observations[-1] is the label of the sequence
            beta, zb = backward((pi, A, O), obs)

            # calculate sums at the desired indexes in the sequence for fractional change
            sum_begin = np.sum(obs[begin[0]:begin[1]]) + obs[begin[0]:begin[1]].count(-1)
            sum_end = np.sum(obs[end[0]:end[1]]) + obs[end[0]:end[1]].count(-1)
            fractional_change = (abs(sum_begin - sum_end)) / sum_begin

            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0, :] * beta[0, :] / za

            for i in range(0, len(obs)):
                # handling missings
                if obs[i] != -1:
                    O1[:, obs[i]] += alpha[i, :] * beta[i, :] / za

            for i in range(1, len(obs)):
                for s1 in range(S):
                    for s2 in range(S):
                        trans = A[s1, s2]
                        # gate mechanism: affect the update by considering fractional_change
                        if s2 == label and fractional_change < threshold:
                            trans = 0
                        if obs[i] != -1:
                            A1[s1, s2] += alpha[i - 1, s1] * trans * O[s2, obs[i]] * beta[i, s2] / za
                        else:
                                A1[s1, s2] += alpha[i - 1, s1] * trans * beta[i, s2] / za

        # normalise pi1, A1, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])

    return pi, A, O


# quick test
a = np.array([[0.6, 0.4], [0.4, 0.6]])
p = np.array([0.7, 0.3])
o = np.array([[0.7, 0.1, 0.2, 0, 0, 0], [0, 0., 0.3, .4, .2, .1]])
label_0, label_1 = 0, 1
# the first two sequences have fractional change higher than threshold and the other two lower
data = [[4, 4, 3, 2, -1, -1, 3, 4, 1, 1, 0, label_0],
        [4, 3, 3, -1, 3, -1, 3, -1, 1, 1, 1, label_0],
        [5, 5, 5, 3, 4, -1, -1, -1, 4, 5, 4, label_1],
        [4, 5, -1, 3, 4, 5, -1, -1, -1, 5, 3, label_1]]


start_prob, transition_prob, emission_prob = ghmm(data, p, a, o, 10,
                                                  threshold=.51, policy=[[0, 2], [-2, -1]], label=label_0)

print(start_prob)
print(transition_prob)
print(emission_prob)
print('\n')

# do perdiction for a new sequence without having label
sequence = [5, 4, -1, 4, 4, 5, -1, -1, -1, 5, 4]
fwd, s = forward((start_prob, transition_prob, emission_prob), sequence)
prob = fwd[len(sequence) - 1, 1] / s

print("prediction probability: {}".format(prob))
print("predicted label: {}".format(1 if prob > .5 else 0))


