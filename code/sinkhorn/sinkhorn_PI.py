import numpy as np
import sinkhorn.utils as utils
from sinkhorn.utils import Matrix2D, Settings
from sinkhorn.utils import evaluate_pi


def update_Q(cost: Matrix2D, pi: Matrix2D, Q: Matrix2D, settings: Settings):
    """ Apply the update rule for the Q-function """
    """
    The arguments are 2D matrices of sizes:
    - |cost| = XxY
    - |pi| = XYxXY
    - |Q| = XYxXY
    """
    # Q_{k+1}(xy,x'y') = c(xy) + gamma * sum_{x",y"} pi_k(x"y"|x'y')*Q_k(x'y', x"y")
    # compute element-wise res1 = pi_k * Q_k
    # aggregate over x"y": res2 = res1 * 1_{xy} (1 * xy)
    # get flatten representation of cost: cost.flatten (xy * 1)
    # extend res2 & flatten cost to be xy * xy
    # Q = extended_res2 + extended_flatten_cost
    # return Q

    # OTHER
    # X, Y = cost.m.shape
    # CF = cost.m.flatten()[:, None].repeat(X * Y, 1)
    # QM = CF + settings.gamma * np.sum(pi.m * Q.m, 1)[None, :].repeat(X * Y, 0)

    # Create COST matrix # ToDo:
    #c = cost.flatten()
    #c = c.repeat(rep_rows=c.rows * c.cols, rep_cols=1)

    # Create pi_k*Q_k matrix
    res1 = pi * Q
    res2 = Matrix2D(res1, res1.shape[0], res1.shape[1])
    res2 = res2.sum_along_cols()
    res2 = res2.repeat(rep_rows=res2.rows * res2.cols, rep_cols=1)
    res2 = res2.transpose()

    # Update Q_{k+1} matrix
    qm = cost.m + settings.gamma * res2.m
    Q = Matrix2D(qm, qm.shape[0], qm.shape[1])

    return Q



def odd_update_policy(pi: Matrix2D, Px: Matrix2D, Q: Matrix2D, settings: Settings):
    """ Apply the update rule for the policy """
    """
    The arguments are 2D matrices of sizes:  
    - |pi| = XYxXY
    - |P| XYxXY (extended versions Px in XxX and Py in YxY)
    - |Q| = XYxXY 
    """
    # Javi pseudocode comments
    # norm(x') = sum_{y"} pi_k(x'y"|xy) * exp(-eta*Q_k(xy,x'y"))
    # pi_{k+1}(x'y'|xy) = pi_k(x'y'|xy) * exp(-eta*Q_k(xy,x'y')) * Px(x'|x) / norm(x')

    # Sergio code version
    X, Y = settings.dimX, settings.dimY
    Pi = pi.m.copy()
    Num = Pi * np.exp(-settings.eta * Q.m)
    Den = np.sum(Num.reshape((X * Y, X, Y)), 2).repeat(Y, 1)
    Pi[Den!=0] = (Num[Den!=0] / Den[Den!=0])
    Pi *= Px.m

    # Update pi from Pi
    pi = Matrix2D(Pi, Pi.shape[0], Pi.shape[1])

    return pi

def even_update_policy(pi: Matrix2D, Py: Matrix2D, Q: Matrix2D, settings: Settings):
    # Javi pseudocode comments
    # norm(y') = sum_{x"} pi_k(x"y'|xy) * exp(-eta*Q_k(xy,x"y'))
    # pi_{k+1}(x'y'|xy) = pi_k(x'y'|xy) * exp(-eta*Q_k(xy,x'y')) * Py(y'|y) / norm(y')

    # Sergio code version
    X, Y = settings.dimX, settings.dimY
    Pi = pi.m.copy()
    Num = Pi * np.exp(-settings.eta * Q.m)
    Den = np.tile(np.sum(Num.reshape((X * Y, X, Y)), 1), (1, X))
    Pi[Den!=0] = (Num[Den!=0] / Den[Den!=0])
    Pi *= Py.m

    # Update pi from Pi
    pi = Matrix2D(Pi, Pi.shape[0], Pi.shape[1])

    return pi

def round(pi, Px: Matrix2D, Py: Matrix2D):

    for x in range(Px.rows):
        for y in range(Py.rows):
            idx = Py.rows * x + y
            pi.m[idx] = utils.round_transpoly(np.reshape(pi.m[idx], (Px.rows, Py.rows)), Px.m[x, :][:, np.newaxis], Py.m[y, :]).flatten()

    return pi


def SPI(Px: Matrix2D, Py: Matrix2D, cost: Matrix2D, settings: Settings):
    pi_0 = utils.get_independent_coupling(Px=Px, Py=Py)
    pi = Matrix2D(pi_0.m)
    Q = Matrix2D()
    #Q.set_2D_matrix(np.zeros((Px.rows * Py.rows, Px.cols * Py.cols)))
    Q.set_rand_2D_matrix(Px.rows * Py.rows, Px.cols * Py.cols) # ToDo: remove this and keep the initial Q_0

    # ToDo:
    X, Y = settings.dimX, settings.dimY
    #  1. create cost matrix (flatten version, ...)
    c = cost.flatten()
    cm = c.repeat(rep_rows=c.rows * c.cols, rep_cols=1)
    #  2. create Px matrix of X*X version
    # PXF = PX.repeat(Y, 0).repeat(Y, 1)
    pxm = Px.m.repeat(Y, 0).repeat(Y, 1)
    Pxm = Matrix2D(pxm, pxm.shape[0], pxm.shape[1])
    #  3. create Py matrix of Y*Y version
    # PYF = np.tile(PY, (X, X))
    pym = np.tile(Py.m, (X, X))
    Pym = Matrix2D(pym, pym.shape[0], pym.shape[1])

    if settings.eta_decay:
        eta_0 = settings.eta
    for k in range(1, settings.K+1):
        #Q.m = np.zeros((Px.rows * Py.rows, Px.cols * Py.cols))
        if settings.epsilon:
            pi_old = pi.m.copy()
        if settings.eta_decay:
            settings.eta =  eta_0 / np.sqrt(k)
        """ Step 1. Update Q matrix """
        for n in range(settings.N):
            Q = update_Q(cost=cm, pi=pi, Q=Q, settings=settings)

        """ Step 2. Update the policy """
        if (k & 1) == 1:  # k is odd
            pi = odd_update_policy(pi=pi, Px=Pxm, Q=Q, settings=settings)
        else:
            pi = even_update_policy(pi=pi, Py=Pym, Q=Q, settings=settings)

        if settings.epsilon:
            if np.allclose(pi.m, pi_old, rtol=settings.epsilon, atol=1e-04):
                print('Converged in ', k, 'iterations')
                break

    if settings.round:
        pi = round(pi, Px, Py)
    # print('distance', distance)
    # distance = 0
    #print(Q.m.reshape(X, Y, X*Y).sum(2))
    return pi

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


