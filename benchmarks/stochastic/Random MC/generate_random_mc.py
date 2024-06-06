from sinkhorn.utils import Matrix2D
import numpy as np
import os
binary = True
for dim in [15]:
    X, Y = dim,dim
    for seed in range(5):
        np.random.seed(seed)
        Px = Matrix2D()
        Px.set_rand_2D_matrix(rows=X, cols=X)
        Px.normalize()
        Py = Matrix2D()
        Py.set_rand_2D_matrix(rows=Y, cols=Y)
        Py.normalize()
        cost = Matrix2D()
        if binary==True:
            cost.set_2D_matrix(np.random.randint(2, size=(X, Y)))
        else:
            cost.set_rand_2D_matrix(rows=X, cols=Y)

        dir = 'Binary_MC_' + str(X) + 'x' + str(Y) + '_seed_' + str(seed)

        os.makedirs(dir)
        np.save(dir + '/Px.npy', Px.m)
        np.save(dir + '/Py.npy', Py.m)
        np.save(dir + '/cost.npy', cost.m)