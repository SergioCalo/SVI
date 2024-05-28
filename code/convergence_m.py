import argparse
import time
import numpy as np
import sinkhorn.utils as utils
from sinkhorn.utils import Matrix2D, Settings
from sinkhorn.sinkhorn_PI import SPI
from sinkhorn.sinkhorn_VI import SVI

import matplotlib.pyplot as plt

def main():
    # Define hyperparameters
    parser = argparse.ArgumentParser(description="MDPOT", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-dx", "--dimensionX", type=int, required=False, default=10)
    parser.add_argument("-dy", "--dimensionY", type=int, required=False, default=10)
    parser.add_argument("-e", "--eta", type=float, required=False, default=10.)
    parser.add_argument("-g", "--gamma", type=float, required=False, default=.95)
    parser.add_argument("-ep", "--epsilon", type=float, required=False, default=None)
    parser.add_argument("-k", "--iterations", type=int, required=False, default=300)
    parser.add_argument("-p", "--projections", type=int, required=False, default=1)
    parser.add_argument("-s", "--seed", type=int, required=False, default=0)
    parser.add_argument("-v", "--verbose", choices=[0, 1, 2], type=int, required=False) #The higher the number the more info you print
    parser.add_argument('--round', default=False, action='store_true')

    args = parser.parse_args()

    #args.seed = 0
    if args.seed is not None:
        np.random.seed(args.seed)

    # Initialize instances
    env_1_path = '../benchmarks/stochastic/4-rooms/p1.json'
    env_2_path = '../benchmarks/stochastic/4-rooms/p0.json'

    Px, Rx, nu_0_x = utils.load_env(env_1_path)
    Py, Ry, nu_0_y = utils.load_env(env_2_path)
    Px.normalize()
    Py.normalize()
    dimX, dimY = Rx.shape[0], Ry.shape[0]
    nu_0 = utils.get_initial_distribution(nu_0_x, nu_0_y)

    cost = Matrix2D()
    cost.set_2D_matrix(M=np.abs(np.subtract.outer(Rx, Ry)))


    eta = 1.
    iters = [750]
    M = [1, 2, 5, 10, 25]

    distances = []
    times = []

    # get the start time
    st = time.time()

    for m in M:
        distances = []
        times = []
        for it in iters:

            settings = Settings(eta=eta, gamma=args.gamma, N=m, K=it, epsilon=args.epsilon, dimX=dimX, dimY=dimY,
                                round=True, eta_decay = False, nu_0=None)
            # get the start time
            st = time.time()

                # Run algorithm
            #pi = SPI(Px=Px, Py=Py, cost=cost, settings=settings)
            pi = SVI(Px=Px, Py=Py, cost=cost, settings=settings)

            pi_0 = utils.get_independent_coupling(Px=Px, Py=Py)
            distance = utils.evaluate_pi(pi, pi_0, cost, settings)

            # Measure total computation time
            et = time.time()
            elapsed_time = et - st
            #print(f'Pi:\n{np.around(pi.m, decimals=3)}')
            #utils.check_constraint_satisfaction(pi, Px, Py)
            distances.append((distance).item(0))
            times.append(elapsed_time)

        plt.plot(iters, distances, label='m: ' + str(m))
        #plt.plot(iters, times, label='m: ' + str(m))

        plt.xlabel("Iterations")
        plt.ylabel("Distance")
        print('m: ', m)
        print('distance: ', distances)
        print('time: ', elapsed_time)
    print(distances)

    plt.legend()
    #plt.ylim(0.00001, 0.1)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.savefig('conv_m.png')
    plt.show()

if __name__ == "__main__":
    main()
