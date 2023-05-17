# %%
import numpy as np
import torch

def HitT(T, transitions : np.array):
    """
    T: temperature
    transitions: matrix of logits
    """	
    # convert each row of logits to probabilities
    transition_matrix = np.zeros(transitions.shape)

    for i in range(len(transitions)):
        row = [x/T if x != 0.0 else -np.infty for x in transitions[i]]
        transition_matrix[i] = torch.softmax(torch.tensor(row), dim=0).tolist()

    '''
    Since the sum of each row is 1, our matrix is row stochastic.
    We'll transpose the matrix to calculate eigenvectors of the stochastic rows.
    '''
    transition_matrix_transp = transition_matrix.T
    eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)

    '''
    Find the indexes of the eigenvalues that are close to one.
    Use them to select the target eigen vectors. Flatten the result.
    '''
    close_to_1_idx = np.isclose(eigenvals,1)
    target_eigenvect = eigenvects[:,close_to_1_idx]
    target_eigenvect = target_eigenvect[:,0]
    # Turn the eigenvector elements into probabilites
    stationary_distrib = target_eigenvect / sum(target_eigenvect)

    # length of matrix
    N = transition_matrix.shape[0]

    # matrix where every row is the stationary distribution
    stationary_distrib_matrix = np.tile(stationary_distrib, (N,1))
    # calculate the fundamental matrix
    fundamental_matrix = np.linalg.inv(np.eye(N,N) - transition_matrix - stationary_distrib_matrix)

    def expected_hittingtime(i, j):
        return (fundamental_matrix[j,j] - fundamental_matrix[i,j])/ stationary_distrib[j]

    all_hitting_times = []
    for i in range(N):
        for j in range(N):
            all_hitting_times.append(expected_hittingtime(i,j))

    return max(all_hitting_times)

def load_logits(logits_FP):
    # read logits from csv

    with open(logits_FP) as f:
        lines = f.readlines()
        states = lines[0].strip().split(",")
        transitions = []
        for line in lines[1:]:
            transitions.append(list(map(float, line.strip().split(","))))
    
    return np.array(transitions), states

def hoeffding_bound(t, n, a, b, HitT):
    """
    t: deviation
    n: number of steps through markov chain
    a: lower bound of image of $f$
    b: upper bound of image of $f$
    HitT: hitting time
    """

    nu2 = 0.25*n*(b-a)*HitT

    return np.exp((-2 * n * (t**2)) / nu2)
# %%
if __name__=="__main__":
    transitions, states = load_logits("../model/logits/logits_vocab2context3.csv")
    print(HitT(0.99, transitions))
# %%
