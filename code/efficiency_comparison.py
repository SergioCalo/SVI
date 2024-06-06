import torch
from ot_markov_distances import discounted_wl_k, discounted_wl_infty
import time
from sinkhorn.sinkhorn_PI import SPI, round
from sinkhorn.sinkhorn_VI import SVI
import utils
from sinkhorn.utils import Matrix2D, Settings
import numpy as np
import pandas as pd

from OTC.entropic_otc import entropic_otc


# Arguments
gamma = 0.75 # discount factor
delta = 1 - gamma
#dims = [2,3,5,10,15,20,25]
dims = [2,3,5,10,15,20,25]

#Algorithm parameters:

#Alg 1
eta_1 = 1.
k_1 = 500
epsilon_1 = 1.e-6

#Alg 2
eta_2 = 1.
k_2 = 500
epsilon_2 = 1.e-6

#O'Connor
L = 100
T = 1000
xi = 200  # xi_vec = [75 100 200];
sink_iter = 100  # sink_iter_vec = [50 100 200];
tau = 0.1  # experiment parameter

#Brugere
max_iter = 1000
sinkhorn_reg= 1/100
sinkhorn_iter = 100
convergence_threshold_rtol = .000001
convergence_threshold_atol = 1e-8


seeds = 5
results = np.zeros((seeds, len(dims), 3))
times = np.zeros((seeds, len(dims), 3))
df = pd.DataFrame(columns=['seed', 'size', 'Alg 1 (Ours) distance', 'Alg 2 (Ours) distance', 'Brugere distance', 'dOTC distance', 'Alg 1 (Ours) time', 'Alg 2 (Ours) time',  'Brugere time', 'dOTC time'])
new_row = {}

load_env = False
for d, dim in enumerate(dims):
    if load_env:
        Px, Rx, nu_0_x = utils.load_env('../benchmarks/stochastic/3_rewards_random/pi_0.2_0.json')
        Py, Ry, nu_0_y = utils.load_env('../benchmarks/stochastic/3_rewards_random/pi_0.2_13.json')
        Px.normalize()
        Py.normalize()
        dimX, dimY = Rx.shape[0], Ry.shape[0]
        nu_0 = utils.get_initial_distribution(nu_0_x, nu_0_y)

        cost = Matrix2D()

        M = np.abs(np.subtract.outer(Rx, Ry))
        cost.set_2D_matrix(M=M)
        X = dimX # Px size
        Y = dimY # Py size

    else:
        nu_0 = None
        X = dim # Px size
        Y = dim # Py size
    for seed in range(seeds):
        new_row['seed'] = seed
        new_row['size'] = dim
        if not load_env:
            dir = '../benchmarks/stochastic/Random MC/MC_' + str(X) + 'x' + str(Y) + '_seed_' + str(seed)
            #print(dir)
            Px = Matrix2D()
            Px.load_2D_matrix(dir=dir + '/Px.npy')
            Py = Matrix2D()
            Py.load_2D_matrix(dir=dir + '/Py.npy')
            cost = Matrix2D()
            cost.load_2D_matrix(dir=dir + '/cost.npy')

        MX = torch.tensor(Px.m, dtype=torch.float).unsqueeze(0)
        MY = torch.tensor(Py.m, dtype=torch.float).unsqueeze(0)
        cost_matrix = torch.tensor(cost.m, dtype=torch.float).unsqueeze(0)

        # Run algorithm 1
        settings = Settings(eta=eta_1, gamma=gamma, N=1, K=k_1, epsilon=epsilon_1, dimX=X, dimY=Y,
                            round=True, eta_decay=False, nu_0=nu_0)
        st = time.time()
        print('Running sinkhorn VI (Algorithm 1): ')
        pi = SVI(Px=Px, Py=Py, cost=cost, settings=settings)
        pi_0 = utils.get_independent_coupling(Px=Px, Py=Py)
        distance = utils.evaluate_pi(pi, pi_0, cost, settings)
        et = time.time()
        elapsed_time = et - st
        print('distance: ', distance.item())
        print(f'[INFO] Total elapsed time: {elapsed_time}')
        new_row['Alg 1 (Ours) distance'] = distance.item(0)
        new_row['Alg 1 (Ours) time'] = elapsed_time

        # Run algorithm 2
        settings = Settings(eta=eta_2, gamma=gamma, N=1, K=k_2, epsilon=epsilon_2, dimX=X, dimY=Y,
                            round=True, eta_decay=True, nu_0=nu_0)
        st = time.time()
        print('Running sinkhorn PI (Algorithm 2): ')
        pi = SPI(Px=Px, Py=Py, cost=cost, settings=settings)
        pi_0 = utils.get_independent_coupling(Px=Px, Py=Py)
        distance = utils.evaluate_pi(pi, pi_0, cost, settings)
        et = time.time()
        elapsed_time = et - st
        print('distance: ', distance.item())
        print(f'[INFO] Total elapsed time: {elapsed_time}')
        new_row['Alg 2 (Ours) distance'] = distance.item(0)
        new_row['Alg 2 (Ours) time'] = elapsed_time
        #print(f'Pi:\n{np.around(pi.m, decimals=3)}')



        # Brugere's
        st = time.time()
        print('Running discounted_wl_infty: ')
        distance, log_P = discounted_wl_infty(MX, MY, distance_matrix=cost_matrix,
                                                         max_iter=max_iter,
                                                         delta = delta,
                                                         sinkhorn_reg=sinkhorn_reg,
                                                         sinkhorn_iter = sinkhorn_iter,
                                                         convergence_threshold_rtol=convergence_threshold_rtol,
                                                         convergence_threshold_atol=convergence_threshold_atol)

        P = torch.exp(log_P).reshape(log_P.shape[1]*log_P.shape[2], log_P.shape[3]*log_P.shape[4]).numpy()
        P = Matrix2D(m=P, rows=P.shape[0], cols=P.shape[1])
        #P = round(P, Px, Py)
        #print(f'Pi:\n{np.around(P.m, decimals=3)}')
        #print('distance: ', distance)
        #print(f'Pi:\n{np.around(P_gamma.m, decimals=3)}')
        distance = utils.evaluate_pi(P, pi_0, cost, settings)
        print('distance: ', distance)

        et = time.time()
        elapsed_time = et - st
        print(f'[INFO] Total elapsed time: {elapsed_time}')
        new_row['Brugere distance'] = distance.item()
        new_row['Brugere time'] = elapsed_time



        # O'connor discounted

        st = time.time()
        print('running OTC: ')


        st = time.time()
        exp_cost, P_gamma, times = entropic_otc(Px.m, Py.m, cost.m, L, T, xi, sink_iter, True, gamma=gamma)
        P_gamma = Matrix2D(m=P_gamma, rows=P_gamma.shape[0], cols=P_gamma.shape[1])
        #print(f'Pi:\n{np.around(P_gamma.m, decimals=3)}')
        distance = utils.evaluate_pi(P_gamma, pi_0, cost, settings)
        et = time.time()
        elapsed_time = et - st
        print(distance)
        print(f'[INFO] Total elapsed time: {elapsed_time}')
        new_row['dOTC distance'] = distance.item()
        new_row['dOTC time'] = elapsed_time
        #print(f'Pi:\n{np.around(P_gamma.m, decimals=3)}')


        # O'connor average
        #distance, P, times = entropic_otc(Px.m, Py.m, cost.m, L, T, xi, sink_iter, True, gamma = None)
        #et = time.time()
        #elapsed_time = et - st
        #print(distance)
        #print(f'[INFO] Total elapsed time: {elapsed_time}')
        #new_row['OTC distance'] = distance.item()
        #new_row['OTC time'] = elapsed_time
        #print(f'Pi:\n{np.around(P, decimals=3)}')

        # Update dataframe
        df.loc[len(df)] = new_row

df.round(4).to_csv(f'results/alg_comparison/efficiency_results_gamma={gamma}.csv', index=False)
df.round(4).to_html(f'results/alg_comparison/efficiency_results_gamma={gamma}.html')
print(df.round(4))