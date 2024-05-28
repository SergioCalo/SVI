import time
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    eta: float
    gamma: float


@dataclass(frozen=False, slots=False)
class Matrix:
    m: np.matrix

    def __init__(self, dim: int, normalized: bool = False):
        self.m = get_rand_matrix(dim=dim)
        if normalized:
            self.m = normalize_matrix(m=self.m)

@dataclass(frozen=False, slots=False)
class Policy:
    pi = np.matrix

    def __init__(self, P: Matrix = None, Q: Matrix = None):
        if(P is not None and Q is not None):
            self.pi = get_independent_coupling(P.m, Q.m)
        else:
            self.pi = None

    def copy(self):
        policy = Policy()
        policy.pi = self.pi.copy()
        return policy

    def evaluate(self, mu_0: np.ndarray, gamma) -> np.ndarray:
        a = (np.eye(self.pi.shape[0], dtype=int) - gamma * self.pi.T)
        b = (1 - gamma) * mu_0
        return np.linalg.solve(a, b)

def initialization(P: Matrix, Q: Matrix, dim: int):
    V_x = np.zeros((dim, dim, dim))
    V_y = np.zeros((dim, dim, dim))
    pi_0 = Policy(P, Q)
    pi = pi_0.copy()
    return pi, pi_0, V_x, V_y


def projection(axis, pi, dim, eta, V, cost, gamma, T, n_projections=50):
    eta_factor = (-1.0/eta)
    for n in range(n_projections):
        for x in range(dim):
            for y in range(dim):
                idx1 = dim*x + y
                for aux in range(dim):
                    if(axis == 'x'):
                        idx2 = dim*aux
                        sumi = sum([pi[idx1, idx2+y_prime] *
                                    np.exp(-eta *
                                           (cost[x, y] + gamma * T[aux] @ V[aux, y_prime]) -
                                           np.log(T[x, aux]))
                                    for y_prime in range(dim)])
                    elif(axis == 'y'):
                        sumi = sum([pi[idx1, dim * (x_prime) + aux] *
                                    np.exp(-eta *
                                           (cost[x, y] + gamma * T[aux] @ V[x_prime, aux]) -
                                           np.log(T[y, aux]))
                                    for x_prime in range(dim)])

                    V[x, y, aux] = eta_factor * np.log(sumi)


def recompute_pi(axis, pi, dim, eta, V_x, V_y, cost, gamma, P, Q, ):
    for x in range(dim):
        for y in range(dim):
            idx1 = dim * (x) + y
            for x_prime in range(dim):
                for y_prime in range(dim):
                    idx2 = dim * (x_prime) + y_prime
                    if axis == 'x':
                        pi[idx1, idx2] = pi[idx1, idx2] * np.exp(
                            -eta * (cost[x, y] + gamma * P[x_prime] @ V_x[x_prime, y_prime] - V_x[x, y, x_prime]))

                    elif axis == 'y':
                        pi[idx1, idx2] = pi[idx1, idx2] * np.exp(
                            -eta * (cost[x, y] + gamma * Q[y_prime] @ V_y[x_prime, y_prime] - V_y[x, y, y_prime]))


def round(pi, P, Q, dim):
    for x in range(dim):
        for y in range(dim):
            idx = dim * (x) + y
            pi[idx] = round_transpoly(np.reshape(pi[idx], (dim, dim)), P[x, :][:, np.newaxis], Q[y, :]).flatten()


def print_results(mu, mu_0, pi, pi_0, cost, verbose):
    if verbose == 2:
        print(f'Pi:\n{np.around(pi, decimals=3)}')
        print(f'Pi sum by columns={pi.sum(1)}')
        print(f'Pi total sum={pi.sum()}')
        print(f"pi_0:\n{np.around(pi_0, decimals=3)}")
        print(f'pi_0 sum by columns={pi_0.sum(1)}')

    if verbose == 1 or verbose == 2:
        c = np.reshape(cost, (pi.shape[0], -1))
        print('Final cost: ', mu @ c)
        print('Independent coupling cost: ', mu_0 @ c)
        # print('cost: ', np.sum(pi @ c) / pi.shape[0])
        # print('Pi_0 cost: ', np.sum(pi_0 @ c) / pi.shape[0])

    if verbose == 0:
        pass


def main():
    # Define hyperparameters
    parser = argparse.ArgumentParser(description="MDPOT", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--dimensions", type=int, required=False, default=2)
    parser.add_argument("-e", "--eta", type=float, required=False, default=1.)
    parser.add_argument("-g", "--gamma", type=float, required=False, default=.95)
    parser.add_argument("-p", "--projections", type=int, required=False, default=5)
    parser.add_argument("-it", "--iterations", type=int, required=False, default=10)
    parser.add_argument("-s", "--seed", type=int, required=False, default=None)
    parser.add_argument("-v", "--verbose", choices=[0, 1, 2], type=int, required=False) #The higher the number the more info you print

    args = parser.parse_args()
    dim = args.dimensions
    eta = args.eta
    gamma = args.gamma
    n_projections = args.projections
    if args.seed is not None:
        np.random.seed(args.seed)

    # Initialize random instances
    P = Matrix(dim=dim, normalized=True)
    Q = Matrix(dim=dim, normalized=True)  # ToDo: allow diff P & Q dimensions
    cost = Matrix(dim=dim, normalized=False)
    policy, policy_0, V_x, V_y = initialization(P, Q, dim)
    mu_0 = stationary_dist(policy_0.pi)


    # get the start time
    st = time.time()

    # Run algorithm
    for n in range(args.iterations):
        projection('x', policy.pi, dim, eta, V_x, cost.m, gamma, P.m, n_projections=n_projections)
        recompute_pi('x', policy.pi, dim, eta, V_x, V_y, cost.m, gamma, P.m, Q.m)
        projection('y', policy.pi, dim, eta, V_y, cost.m, gamma, Q.m, n_projections=n_projections)
        recompute_pi('y', policy.pi, dim, eta, V_x, V_y, cost.m, gamma, P.m, Q.m )

    # Measure total computation time
    et = time.time()
    elapsed_time = et - st
    print(f'[INFO] Total elapsed time: {elapsed_time}')
    mu = policy.evaluate(mu_0, gamma)


    """
    Print the results (pi and cost and compare with independent) before and after applying 
    the rounding algorithm (by Altschuler et al). This algorithm ensures the final coupling 
    to verify the coupling constraints, since our algorithm might only do it approximately
    """
    print('-----BEFORE ROUNDING-----')
    print_results(mu, mu_0, policy.pi, policy_0.pi, cost.m, args.verbose)
    round(policy.pi, P.m, Q.m, dim)
    mu = policy.evaluate(mu_0, gamma)
    print('-----AFTER ROUNDING-----')
    print_results(mu, mu_0, policy.pi, policy_0.pi, cost.m, args.verbose)


if __name__ == "__main__":
    main()
