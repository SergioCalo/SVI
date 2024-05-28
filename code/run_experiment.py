import numpy as np
import sinkhorn.utils as utils
from sinkhorn.utils import Matrix2D, Settings
from sinkhorn.sinkhorn_PI import SPI
from sinkhorn.sinkhorn_VI import SVI
import argparse
import os


def run(env_1_path: str, env_2_path: str):
    Px, Rx, nu_0_x = utils.load_env(env_1_path)
    Py, Ry, nu_0_y = utils.load_env(env_2_path)
    Px.normalize()
    Py.normalize()
    dimX, dimY = Rx.shape[0], Ry.shape[0]
    nu_0 = utils.get_initial_distribution(nu_0_x, nu_0_y)

    cost = Matrix2D()
    cost.set_2D_matrix(M=np.abs(np.subtract.outer(Rx, Ry)))

    settings = Settings(eta=5., gamma=0.95, N=1, K=50, epsilon=1e-8, dimX=dimX, dimY=dimY, round=True, eta_decay=True, nu_0=nu_0)
    pi = SVI(Px=Px, Py=Py, cost=cost, settings=settings)
    # pi = SPI(Px=Px, Py=Py, cost=cost, settings=settings)
    pi_0 = utils.get_independent_coupling(Px=Px, Py=Py)
    distance_occ_coupling = utils.evaluate_pi(pi, pi_0, cost, settings)

    settings = Settings(eta=5., gamma=0.95, N=1, K=500, epsilon=1e-10, dimX=dimX, dimY=dimY, round=True, eta_decay=True, nu_0=nu_0)
    x_0 = nu_0_x.nonzero()[0]
    y_0 = nu_0_y.nonzero()[0]

    V = utils.V_function(pi=pi, cost=cost, settings=settings)
    distance_V_initial_state = V[x_0,y_0]
    distance_expect_V = utils.expect_V_nu(pi, pi_0, cost, settings)

    #trajectory = utils.sample_trajectory(pi, nu_0=(0,39), h=12, settings=settings)
    #print(trajectory)

    return pi, distance_V_initial_state, distance_expect_V, distance_occ_coupling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDPOT-Experiment", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-f", "--folder", type=str, required=False, default=None)
    parser.add_argument("-f1", "--file_1", type=str, required=False, default="../benchmarks/deterministic/miconic/p01.json")
    parser.add_argument("-f2", "--file_2", type=str, required=False, default="../benchmarks/deterministic/miconic/p03.json")
    parser.add_argument("-s", "--save_dir", type=str, required=False, default="./")

    args = parser.parse_args()
    if args.folder:
        instances = sorted(os.listdir(args.folder))
        print(instances)
        distance_matrix_V_initial_state = np.zeros((len(instances),len(instances)))
        distance_matrix_expect_V = np.zeros((len(instances), len(instances)))
        distance_matrix_occ_coupling = np.zeros((len(instances), len(instances)))
        for idx_1, filename_1 in enumerate(instances):
            for idx_2, filename_2 in enumerate(instances):

                f1 = os.path.join(args.folder, filename_1)
                f2 = os.path.join(args.folder, filename_2)

                if os.path.isfile(f1):
                    print(idx_1)
                if os.path.isfile(f2):
                    print(idx_2)
                _, distance_V_initial_state, distance_expect_V, distance_occ_coupling = run(env_1_path=f1, env_2_path=f2)
                print(distance_V_initial_state)
                distance_matrix_V_initial_state[idx_1, idx_2] = distance_V_initial_state
                distance_matrix_expect_V[idx_1, idx_2] = distance_expect_V
                distance_matrix_occ_coupling[idx_1, idx_2] = distance_occ_coupling
                print(distance_occ_coupling)

        np.savetxt(f'{args.save_dir}/distance_matrix_V_initial_state.txt', distance_matrix_V_initial_state, fmt='%1.4f')
        np.savetxt(f'{args.save_dir}/distance_matrix_expect_V.txt', distance_matrix_expect_V, fmt='%1.4f')
        np.savetxt(f'{args.save_dir}/distance_matrix_occ_coupling.txt', distance_matrix_occ_coupling, fmt='%1.4f')


    else:
        f1 = args.file_1
        f2 = args.file_2
        pi, distance_V_initial_state, distance_expect_V, distance_occ_coupling = run(env_1_path=f1, env_2_path=f2)
        print('V[x0,y0] = ', distance_V_initial_state.item())
        print('<mu,c> = ', distance_occ_coupling.item())
        print('E[V] = ', distance_expect_V)


