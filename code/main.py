import argparse
import time
import numpy as np

import sinkhorn.utils as utils
from sinkhorn.utils import Matrix2D, Settings
from sinkhorn.sinkhorn_PI import SPI
from sinkhorn.sinkhorn_VI import SVI

def main():
    # Define hyperparameters
    parser = argparse.ArgumentParser(description="MDPOT", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-dx", "--dimensionX", type=int, required=False, default=3)
    parser.add_argument("-dy", "--dimensionY", type=int, required=False, default=3)
    parser.add_argument("-e", "--eta", type=float, required=False, default=1.)
    parser.add_argument("-g", "--gamma", type=float, required=False, default=.95)
    parser.add_argument("-ep", "--epsilon", type=float, required=False, default=None)
    parser.add_argument("-k", "--iterations", type=int, required=False, default=20)
    parser.add_argument("-p", "--projections", type=int, required=False, default=1)
    parser.add_argument("-s", "--seed", type=int, required=False, default=None)
    parser.add_argument("-v", "--verbose", choices=[0, 1, 2], type=int, required=False) #The higher the number the more info you print
    parser.add_argument('--round', default=False, action='store_true')


    args = parser.parse_args()
    dimX = args.dimensionX
    dimY = args.dimensionY
    settings = Settings(eta=args.eta, gamma=args.gamma, N=args.projections, K=args.iterations, epsilon=args.epsilon, dimX=dimX, dimY=dimY, round=args.round, eta_decay=True, nu_0=None)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Initialize random instances
    Px = Matrix2D()
    Px.set_rand_2D_matrix(rows=dimX, cols=dimX)
    Px.normalize()
    Py = Matrix2D()
    Py.set_rand_2D_matrix(rows=dimY, cols=dimY)
    Py.normalize()
    cost = Matrix2D()
    cost.set_rand_2D_matrix(rows=dimX, cols=dimY)
    cost.normalize()

    # get the start time
    st = time.time()

    # Run algorithm
    # pi = SPI(Px=Px, Py=Py, cost=cost, settings=settings)
    pi = SVI(Px=Px, Py=Py, cost=cost, settings=settings)
    pi_0 = utils.get_independent_coupling(Px=Px, Py=Py)
    distance = utils.evaluate_pi(pi, pi_0, cost, settings)

    # Measure total computation time
    et = time.time()
    elapsed_time = et - st
    print(f'[INFO] Total elapsed time: {elapsed_time}')
    print(f'Pi:\n{np.around(pi.m, decimals=3)}')
    print(f'Total distance={distance}')
    #print(utils.sample_trajectory(pi, nu_0,  h=20, settings=settings))

if __name__ == "__main__":
    main()
