import numpy as np
import time
import sys

def logsumexp(X, dim):
    """
    Compute log(sum(exp(X),dim)) while avoiding numerical underflow.
    By default dim = 1 (columns).
    Written by Mo Chen (sth4nth@gmail.com).
    ref: https://github.com/oconnor-kevin/OTC/blob/main/ot_algorithms/logsumexp.m
    [NEW]: dim is always filled, so no need for the first if
    """
    
    #y = max(X, [], dim)
    #s = y+log(sum(exp(bsxfun(@minus,X,y)),dim))
    # i = isinf(y)
    # if any(i(:))

    d1, d2 = X.shape
    y = np.max(X, axis=dim)
    repeated_y = np.repeat([y],d1*d2/np.size(y),axis=0)
    if dim == 1:
        repeated_y = repeated_y.T
    s = y + np.log(np.sum(np.exp(np.subtract(X,repeated_y)), dim)) # ToDo

    i = np.where(np.isinf(y))[0]
    if i.any():
        s[i] = y[i]  # ToDo: check if it needs to be copied
    return s


def round_transpoly(X, r, c):
    """
    Implementation of our algorithm for rounding a matrix onto the U_{r,c}
    transport polytope. See Section 2 of the paper for details.
    ref: https://github.com/oconnor-kevin/OTC/blob/main/ot_algorithms/round_transpoly.m
    """
    A = X.copy()
    n1, n2 = A.shape
    r_A = np.sum(A,axis=1)    
    for i in range(n1):
        scaling = min(1, r[i]/r_A[i])
        A[i,:] = scaling * A[i,:]
    
    c_A = np.sum(A,axis=0)
    for j in range(n2):
        scaling = min(1, c[j]/c_A[j])
        A[:,j] = scaling * A[:,j]
    
    r_A = np.sum(A, axis=1)
    c_A = np.sum(A, axis=0)
    err_r = r_A - r
    err_c = c_A - c
    if (np.size(np.nonzero(err_r)[0])>0) and (np.size(np.nonzero(err_c)[0])>0):
        A = A + np.outer(err_r, err_c) / np.sum(np.absolute(err_r))
    
    return A


def logsinkhorn(A,r,c,T):
    """
    %% 
    % Implementation of classical Sinkhorn algorithm for matrix scaling.
    % Each iteration simply alternately updates (projects) all rows or
    % all columns to have correct marginals.
    % 
    % Input parameters:
    %  -- A:  -xi*C
    %  -- r:  desired row sums (marginals)         (dims: nx1)
    %  -- c:  desired column sums (marginals)      (dims: 1xn)
    %  -- T:  number of full Sinkhorn iterations (normalize ALL row or cols)
    %  -- C:  cost matrix for OT
    %
    % Output:
    %  -- P:   final scaled matrix
    ref: https://github.com/oconnor-kevin/OTC/blob/main/ot_algorithms/logsinkhorn.m
    """
    #r = r[np.newaxis].T


    dx, dy = A.shape
    f = np.zeros((dx,1))
    g = np.zeros((1,dy))
    
    for t in range(1, T+1):
        if (t%2) == 1:
            # rescale rows
            f = np.log(r) - logsumexp(A+g, 1)
        else:
            # rescale columns
            g = np.log(c) - logsumexp(A+f[np.newaxis].T, 0)
    P = round_transpoly(np.exp(f[np.newaxis].T+A+g), r, c)
    return P


def get_ind_tc(Px, Py):
    """
    Compute independent coupling between Px & Py
    ref: https://github.com/oconnor-kevin/OTC/blob/main/get_ind_tc.m
    """
    dx, dx_col = Px.shape
    dy, dy_col = Py.shape
    
    P_ind = np.zeros((dx*dy, dx_col*dy_col))
    
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy*x_row + y_row
                    idx2 = dy_col*x_col + y_col
                    P_ind[idx1, idx2] = Px[x_row, x_col] * Py[y_row, y_col]    
    
    return P_ind


def approx_tce(P, c, L, T):
    """
    Approximate transition coupling evaluation
    ref: https://github.com/oconnor-kevin/OTC/blob/main/approx_tce.m
    """
    d = P.shape[0]
    # c = np.reshape(np.transpose(c), (d,1)).copy()
    c = np.reshape(c, (d,1)).copy() #NEW

    c_max = c.max()
    g_old = c
    #g = P * g_old
    g = P @ g_old # NEW
    l = 1
    tol = 1e-12
    while (l<=L) and (np.absolute(g-g_old).max() > tol*c_max):
        g_old = g.copy() # NEW
        #g = P * g_old
        g = P @ g_old # NEW
        l = l+1
    
    g = np.mean(g) * np.ones((d,1))

    diff = c - g
    h = diff
    t = 1
    while (t<=T) and (np.absolute(P@diff).max() > tol*c_max): # NEW
        h = h + P@diff
        diff = P@diff
        t = t+1
    return g, h


def entropic_tci(h, P0, Px, Py, xi, sink_iter):
    """
    Entropic transition coupling improvement.
    ref: https://github.com/oconnor-kevin/OTC/blob/main/entropic_tci.m
    """
    dx = Px.shape[0]
    dy = Py.shape[0]
    P = P0.copy()
    
    # Try to improve with respect to h.
    #print(f"{h=}")
    # h_mat = np.transpose(np.reshape(h, (dy, dx))).copy()
    h_mat = np.reshape(h, (dy, dx)).copy() #NEW

    K = -xi * h_mat
    #print(f"{K=}")
    #print(f"Pre {P=}")
    for i in range(dx):
        for j in range(dy):
             # Run Sinkhorn on each pair of rows, taking care to ignore zeros.
             dist_x = Px[i,:]
             dist_y = Py[j,:]
             x_idxs = np.nonzero(dist_x)[0]
             y_idxs = np.nonzero(dist_y)[0]
             idxs = np.array(np.meshgrid(x_idxs,y_idxs)).T.reshape(-1,2)
             if (np.size(x_idxs) == 1) or (np.size(y_idxs)==1):
                #print(f"{i=};{j=}")
                P[dy*i + j,:] = P0[dy*i + j,:].copy()
             else:
                sol = logsinkhorn(K[x_idxs, :][:, y_idxs], dist_x[x_idxs].T, dist_y[y_idxs], sink_iter)
                #sol = logsinkhorn(K[idxs[:,0], idxs[:,1]], np.transpose(dist_x[x_idxs]), dist_y[y_idxs], sink_iter)
                #print(f"{sol=}")
                sol_full = np.zeros((dx,dy), dtype=float)
                #print(f"{x_idxs=}; {type(x_idxs)=}")
                #sol_full[x_idxs, :][:, y_idxs] = sol
                #print(f"{idxs[:,0]=}; {idxs[:,1]=}")
                #print(f"{sol_full[idxs[:,0], idxs[:,1]]=}")
                #print(f"{sol.flatten()=}")
                sol_full[idxs[:,0], idxs[:,1]] = sol.flatten()
                #print(np.reshape(np.transpose(sol_full), (1,dx*dy)).copy())
                #print(np.reshape(sol_full, (1,dx*dy)).copy())

                #sys.exit()
                P[dy*i+j,:] = np.reshape(sol_full, (1,dx*dy)).copy()  # ToDo: is the copy required here?
    
    
    #print(f"Post {P=}")
    
    return P


def discounted_approx_tce(P, c, L, T, gamma):
    """
    Approximate transition coupling evaluation
    ref: https://github.com/oconnor-kevin/OTC/blob/main/approx_tce.m
    """

    d = P.shape[0]
    # c = np.reshape(np.transpose(c), (d,1)).copy()
    c = np.reshape(c, (d, 1)).copy()  # NEW
    c_max = c.max()
    tol = 1e-12
    discount = gamma * P

    #h = np.zeros(c.shape)
    d = c
    h = c

    t = 1
    while (t <= T) and (np.absolute(P @ c).max() > tol * c_max):  # NEW
        #h = h + discount * P @ c
        #h = P @ (h + gamma * c)
        h = h + P @ d
        d = gamma * (P @ d)
        t = t + 1
        #discount *= gamma
        #print(discount)

        '''diff = c - g
        h = diff
        t = 1
        while (t <= T) and (np.absolute(P @ diff).max() > tol * c_max):  # NEW
            h = h + P @ diff
            diff = P @ diff'''
    return h



def discounted_entropic_tci(h, P0, Px, Py, xi, sink_iter):
    """
    Entropic transition coupling improvement.
    ref: https://github.com/oconnor-kevin/OTC/blob/main/entropic_tci.m
    """
    dx = Px.shape[0]
    dy = Py.shape[0]
    P = P0.copy()

    # Try to improve with respect to h.
    # print(f"{h=}")
    # h_mat = np.transpose(np.reshape(h, (dy, dx))).copy()
    h_mat = np.reshape(h, (dy, dx)).copy()  # NEW

    K = -xi * h_mat
    # print(f"{K=}")
    # print(f"Pre {P=}")
    for i in range(dx):
        for j in range(dy):
            # Run Sinkhorn on each pair of rows, taking care to ignore zeros.
            dist_x = Px[i, :]
            dist_y = Py[j, :]
            x_idxs = np.nonzero(dist_x)[0]
            y_idxs = np.nonzero(dist_y)[0]
            idxs = np.array(np.meshgrid(x_idxs, y_idxs)).T.reshape(-1, 2)
            if (np.size(x_idxs) == 1) or (np.size(y_idxs) == 1):
                # print(f"{i=};{j=}")
                P[dy * i + j, :] = P0[dy * i + j, :].copy()
            else:
                sol = logsinkhorn(K[x_idxs, :][:, y_idxs], dist_x[x_idxs].T, dist_y[y_idxs], sink_iter)
                # sol = logsinkhorn(K[idxs[:,0], idxs[:,1]], np.transpose(dist_x[x_idxs]), dist_y[y_idxs], sink_iter)
                # print(f"{sol=}")
                sol_full = np.zeros((dx, dy), dtype=float)
                # print(f"{x_idxs=}; {type(x_idxs)=}")
                # sol_full[x_idxs, :][:, y_idxs] = sol
                # print(f"{idxs[:,0]=}; {idxs[:,1]=}")
                # print(f"{sol_full[idxs[:,0], idxs[:,1]]=}")
                # print(f"{sol.flatten()=}")
                sol_full[idxs[:, 0], idxs[:, 1]] = sol.flatten()
                # print(np.reshape(np.transpose(sol_full), (1,dx*dy)).copy())
                # print(np.reshape(sol_full, (1,dx*dy)).copy())

                # sys.exit()
                P[dy * i + j, :] = np.reshape(sol_full, (1, dx * dy)).copy()  # ToDo: is the copy required here?

    return P
def entropic_otc(Px, Py, c, L, T, xi, sink_iter, time_iters, gamma=None):
    """
    Entropic transition coupling iteration.
    ref: https://github.com/oconnor-kevin/OTC/blob/main/entropic_otc.m
    """
    dx = Px.shape[0] # get first dimension
    dy = Py.shape[0]  # get first dimension
    max_c = c.max() # get max element in cost matrix
    tol = 1e-5 * max_c # tolerance
    
    g_old = max_c * np.ones((dx*dy, 1), dtype=float) # |g_old| = XYx1
    g = g_old - 10.*tol
    exp_cost = 0.
    P = get_ind_tc(Px, Py) # compute independent coupling
    times = []
    iter_ctr = 0
    tic, toc = 0., 0.  # start - end time
    
    #while g_old[0] - g[0] > tol:
    for it in range(100):
        #print(g_old[0] - g[0])
        iter_ctr = iter_ctr + 1
        #print(f"EntropicOTC Iteration: {iter_ctr}\n")
        P_old = P.copy()
        g_old = g.copy()
        
        if time_iters:
            tic = time.time()
            
        # Approximate transition coupling evaluation
        if gamma:
            h = discounted_approx_tce(P, c, L, T, gamma)
        else:
            g, h = approx_tce(P, c, L, T)

        exp_cost = g[0]
        #print(f"{exp_cost=}")  # display exp_cost
        
        # Entropic transition coupling improvement
        if gamma:
            P = discounted_entropic_tci(h, P_old, Px, Py, xi, sink_iter)
        else:
            P = entropic_tci(h, P_old, Px, Py, xi, sink_iter)
        
        if time_iters:
            toc = time.time()
            times.append([toc-tic])
            
    return exp_cost, P, times


if __name__ == "__main__":
    """
    Experiment setting (except for d=2) taken from:
    https://github.com/oconnor-kevin/OTC/blob/main/run_time_experiment.m
    """
    np.random.seed(0)    
    # Algorithm parameters
    L = 100;
    T = 1000;
    xi = 75  # xi_vec = [75 100 200];
    sink_iter = 2000  # sink_iter_vec = [50 100 200];
    
    tau = 0.1  # experiment parameter
    
    # Simulate marginals and cost.    
    d = 2 # size of all dimensions
    c = np.absolute(np.random.normal(0, 1., (d, d)))
    c = np.divide(c, c.max())
        
    Px = np.random.normal(0, 1., (d, d))
    Px = np.exp(tau*Px) / np.sum(np.exp(tau*Px),axis=1)  # ToDo: check that this division is equivalent to Matlab's division
    Py = np.random.normal(0, 1., (d, d))
    Py = np.exp(tau*Py) / np.sum(np.exp(tau*Py),axis=1)
    print(f"{Py=}\n{np.exp(tau*Py)=}\n{np.sum(np.exp(tau*Py),axis=1)=}")

    print(f"{c=};\n{Px=};\n{Py=};\n")
    
    print(f"{get_ind_tc(Px, Py)=}")
    
    
    # Run Entropic OTC
    exp_cost, P, times = entropic_otc(Px, Py, c, L, T, xi, sink_iter, True)
    print(f"{exp_cost=};\n{P=};\n{times=};\n")
    
