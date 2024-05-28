import numpy as np
import sinkhorn.utils as utils
from sinkhorn.utils import Matrix2D, Settings
from sinkhorn.utils import evaluate_pi

def compute_Q(X, Y, CF, gamma, Pr, Q_old):
	return CF + gamma * np.sum(Pr * Q_old, 1)[None,:].repeat(X * Y, 0)

def update_Vx(CF, pi: Matrix2D, PXX, VX: Matrix2D, settings: Settings):
    """ Apply the update rule for the Q-function """
    """
    The arguments are 2D matrices of sizes:
    - |cost| = XxY
    - |pi| = XYxXY
    - |Q| = XYxXY
    """
    X, Y = settings.dimX, settings.dimY
    QX = CF.m + settings.gamma * np.sum(PXX * VX, 1)[None, :].repeat(X * Y, 0)
    Prod = (pi.m * np.exp(-settings.eta * QX)).reshape((X * Y, X, Y))
    #VX = -np.log(np.sum(Prod, 2) / PXX.m) / settings.eta
    in_log = np.divide(np.sum(Prod, 2), PXX.m, out=np.ones_like(np.sum(Prod, 2)), where=PXX.m != 0)
    #print(in_log)
    VX = -np.log(in_log) / settings.eta

    return Matrix2D(QX), Matrix2D(VX)

def update_Vy(CF, pi: Matrix2D, PYY, VY: Matrix2D, settings: Settings):
    """ Apply the update rule for the Q-function """
    """
    The arguments are 2D matrices of sizes:
    - |cost| = XxY
    - |pi| = XYxXY
    - |Q| = XYxXY
    """
    X, Y = settings.dimX, settings.dimY
    QY = CF.m + settings.gamma * np.sum(PYY * VY, 1)[None, :].repeat(X * Y, 0)
    Prod = (pi.m * np.exp(-settings.eta * QY)).reshape((X * Y, X, Y))
    # VY = -np.log(np.sum(Prod, 1) / PYY.m) / settings.eta
    in_log = np.divide(np.sum(Prod, 1), PYY.m, out=np.ones_like(np.sum(Prod, 1)), where=PYY.m != 0)
    VY = -np.log(in_log) / settings.eta
    return Matrix2D(QY), Matrix2D(VY)


def even_update_policy(pi: Matrix2D, QX, VX, settings: Settings):
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
    pi = pi.m * np.exp(-settings.eta * (QX.m - VX.m.repeat(Y, 1)))

    return Matrix2D(pi, pi.shape[0], pi.shape[1])

def odd_update_policy(pi: Matrix2D, QY, VY, settings: Settings):
    # Javi pseudocode comments
    # norm(y') = sum_{x"} pi_k(x"y'|xy) * exp(-eta*Q_k(xy,x"y'))
    # pi_{k+1}(x'y'|xy) = pi_k(x'y'|xy) * exp(-eta*Q_k(xy,x'y')) * Py(y'|y) / norm(y')

    # Sergio code version
    X, Y = settings.dimX, settings.dimY
    pi = pi.m * np.exp(-settings.eta * (QY.m - np.tile(VY.m, (1, X))))

    return Matrix2D(pi, pi.shape[0], pi.shape[1])

def round(pi, Px: Matrix2D, Py: Matrix2D):

    for x in range(Px.rows):
        for y in range(Py.rows):
            idx = Py.rows * x + y
            pi.m[idx] = utils.round_transpoly(np.reshape(pi.m[idx], (Px.rows, Py.rows)), Px.m[x, :][:, np.newaxis], Py.m[y, :]).flatten()

    return pi


def SVI(Px: Matrix2D, Py: Matrix2D, cost: Matrix2D, settings: Settings):
    pi_0 = utils.get_independent_coupling(Px=Px, Py=Py)
    X, Y = settings.dimX, settings.dimY

    PX = Px.m
    PY = Py.m
    c = cost.flatten()

    VX = Matrix2D(np.random.random((X * Y, X)))  # action-values corresponding to VX
    VY = Matrix2D(np.random.random((X * Y, Y)))  # action-values corresponding to VY
    pi = Matrix2D(pi_0.m)  # initial policy

    # create useful matrices
    CF = Matrix2D(c.m.repeat(X * Y, 1))  # (X * Y) x (X * Y) version of c
    PXX = Matrix2D(PX.repeat(Y, 0))  # (X * Y) x X version of PX
    PYY = Matrix2D(np.tile(PY, (X, 1)))  # (X * Y) x Y version of PY

    if settings.eta_decay:
        eta_0 = settings.eta
    for k in range(1, settings.K+1):
        if settings.epsilon:
            pi_old = pi.m.copy()

        if settings.eta_decay:
            settings.eta =  eta_0 / np.sqrt(k)
        """ Step 1. Update Q matrix """
        if (k & 1) == 1:  # k is odd
            for n in range(settings.N):
                QY, VY = update_Vy(CF=CF, pi=pi, PYY=PYY, VY=VY, settings=settings)
                # print(VY)
        else:
            for n in range(settings.N):
                QX, VX = update_Vx(CF=CF, pi=pi, PXX=PXX, VX=VX, settings=settings)


        """ Step 2. Update the policy """
        if (k & 1) == 1:  # k is odd
            pi = odd_update_policy(pi=pi, QY=QY, VY=VY, settings= settings)
        else:
            pi = even_update_policy(pi=pi, QX=QX, VX=VX, settings= settings)

        if settings.epsilon:
            if np.allclose(pi.m, pi_old, rtol=settings.epsilon, atol=1e-04):
                print('Converged in ', k, 'iterations')
                break
    if settings.round:
        pi = round(pi, Px, Py)
    #distance = evaluate_pi(pi, pi_0, c, settings)
    #distance = 0
    return pi