from scipy.io import savemat
import numpy as np
import glob
import os


for dim in [2,3,5,10,15,20,25]:
    X, Y = dim,dim
    for seed in range(5):
        dir = 'MC_' + str(X) + 'x' + str(Y) + '_seed_' + str(seed)
        npzFiles = glob.glob(dir + "/*.npy")
        for f in npzFiles:
            fm = os.path.splitext(f)[0]+'.txt'
            d = np.load(f)
            np.savetxt(fm, d)
            print('generated ', fm, 'from', f)