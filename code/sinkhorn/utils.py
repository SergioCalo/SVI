import numpy as np
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings("ignore")

@dataclass(frozen=False)
class Settings:
    eta: float  # entropy factor
    gamma: float  # discount factor
    epsilon: float  # converegence criterion
    N: int  # num of projections
    K: int  # num of sinkhorn iterations
    dimX: int
    dimY: int
    round: bool
    eta_decay: float
    nu_0: np.array


@dataclass(frozen=False, slots=False)
class Matrix2D:
    m: np.array = field(default_factory=lambda: np.array(''))
    rows: int = 1
    cols: int = 0

    def set_rand_2D_matrix(self, rows: int, cols: int):
        self.m = get_rand_2D_matrix(rows=rows, cols=cols)
        self.rows = rows
        self.cols = cols

    def set_2D_matrix(self, M: np.array):
        self.m = M.copy()
        self.rows, self.cols = self.m.shape

    def load_2D_matrix(self, dir: str):
        self.m = np.load(dir, allow_pickle=True)
        self.rows, self.cols = self.m.shape

    def load_2D_matrix_from_json(self, data):
        self.m = np.array(data, float)
        self.rows, self.cols = self.m.shape

    def normalize(self):
        self.m = normalize_matrix(m=self.m)

    def repeat(self, rep_rows: int, rep_cols: int):
        m = np.repeat(self.m, rep_cols, axis=0)
        m = np.repeat(m, rep_rows, axis=1)
        return Matrix2D(m, m.shape[0], m.shape[1])

    def tile(self, t_rows: int, t_cols: int):
        m = np.tile(self.m, (t_rows, t_cols))
        return Matrix2D(m, m.shape[0], m.shape[1])

    def flatten(self):
        m = self.m.flatten()[:, None]
        rows, cols = self.m.shape
        return Matrix2D(m, rows, cols)

    def sum_along_rows(self):
        """ Produces one row after adding all elements column-wise """
        m = np.sum(self.m, axis=0)[None, :]
        rows, cols = m.shape
        return Matrix2D(m, rows, cols)

    def sum_along_cols(self):
        """ Produces one column after adding all elements row-wise """
        m = np.sum(self.m, axis=1)[:, None]
        rows, cols = m.shape
        return Matrix2D(m, rows, cols)

    def transpose(self):
        m = self.m.T
        rows, cols = m.shape
        return Matrix2D(m, rows, cols)

    def __mul__(self, other):
        return np.multiply(self.m, other.m)  # element-wise multiplication

    def __add__(self, other):
        return np.add(self.m, other.m)  # element-wise addition

    def __sub__(self, other):
        return np.subtract(self.m, other.m)  # element-wise addition


def get_independent_coupling(Px: Matrix2D, Py: Matrix2D) -> Matrix2D:
    """ Compute the independent coupling of two transition kernels """
    """ Assumption: Px & Py are square matrices """
    rPx = Px.repeat(rep_rows=Py.rows, rep_cols=Py.cols)
    tPy = Py.tile(t_rows=Px.rows, t_cols=Px.cols)
    return Matrix2D(rPx * tPy, Px.rows * Py.rows, Px.cols * Py.cols)


def get_initial_distribution(x_0, y_0):
    dimX, dimY = x_0.shape[0], y_0.shape[0]
    nu_0 = np.zeros((dimX, dimY))
    for x in range(dimX):
        for y in range(dimY):
            nu_0[x, y] = x_0[x] * y_0[y]
    return nu_0.flatten()


def stationary_dist(m: np.array) -> np.ndarray:
    eigen_vals, eigen_vecs = np.linalg.eig(m.T)
    eigen_vec_1 = eigen_vecs[:, np.isclose(eigen_vals, 1)][:, 0]
    return (eigen_vec_1 / eigen_vec_1.sum()).real


def get_rand_2D_matrix(rows: int, cols: int) -> np.array:
    return np.random.rand(rows * cols).reshape((rows, cols))


def normalize_matrix(m: np.array) -> np.array:
    row_sums = m.sum(axis=1)
    m /= row_sums[:, np.newaxis]
    return m


def round_transpoly(X, r, c):
    A = X.copy()
    n1, n2 = A.shape
    r_A = np.sum(A, axis=1)

    for i in range(n1):
        scaling = min(1, r[i] / r_A[i])
        A[i, :] = scaling * A[i, :]

    c_A = np.sum(A, axis=0)
    for j in range(n2):
        scaling = min(1, c[j] / c_A[j])
        A[:, j] = scaling * A[:, j]

    r_A = np.sum(A, axis=1)[:, np.newaxis]
    c_A = np.sum(A, axis=0)
    err_r = r_A - r
    err_c = c_A - c

    if not np.all(err_r == 0) and not np.all(err_c == 0):
        A = A + np.outer(err_r, err_c) / np.sum(np.abs(err_r))

    return A


def compute_mu(nu, pi):
    mu = np.zeros(pi.shape)
    for xy in range(pi.shape[0]):
        for xy_prime in range(pi.shape[1]):
            mu[xy, xy_prime] = nu[xy] * pi[xy, xy_prime]
    return mu


def check_constraint_satisfaction(Pi, Px, Py):
    # print('Constraint satisfaction:')
    nu = stationary_dist(Pi.m)
    mu = compute_mu(nu, Pi.m)

    # print('sum over y x_prime y_prime of mu: ', mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(1))
    # print('nu_x: ', stationary_dist(Px.m))
    # print('sum over x x_prime y_prime of mu: ', mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(0))
    # print('nu_y: ', stationary_dist(Py.m))
    nu_prime_x = mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(1)
    nu_prime_y = mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(0)
    nu_x = stationary_dist(Px.m)
    nu_y = stationary_dist(Py.m)
    if not np.allclose(nu_prime_x, nu_x, rtol=1e-9) or not np.allclose(nu_prime_y, nu_y, rtol=1e-9):
        print("FAILED CONSTRAINT SATISFACTION TEST")


def evaluate_pi(pi: Matrix2D, pi_0: Matrix2D, cost: Matrix2D, settings: Settings):
    if settings.nu_0 is None:
        nu_0 = stationary_dist(pi_0.m)
    else:
        nu_0 = settings.nu_0
    a = (np.eye(pi.rows, dtype=int) - settings.gamma * pi.transpose().m)
    b = (1 - settings.gamma) * nu_0
    nu = np.linalg.solve(a, b)
    c = np.reshape(cost.m, (pi.rows, -1))
    distance = nu @ c
    return distance


def read_json(dir_filename: str):
    with open(dir_filename, 'r') as json_file:
        return json.load(json_file)


def load_env(dir_filename: str):
    data = read_json(dir_filename=dir_filename)
    P = Matrix2D()
    P.load_2D_matrix_from_json(data['transition_function'])
    P.normalize()
    R = np.array(data['reward'], float)
    nu_0 = np.array(data['nu_0'], float)
    return P, R, nu_0


def V_function(pi: Matrix2D, cost: Matrix2D, settings: Settings):
    dimX, dimY = settings.dimX, settings.dimY
    val = (np.linalg.inv(
        np.eye(dimX * dimY) - settings.gamma * pi.m) @ cost.m.flatten()).reshape(dimX, dimY)
    return val

def expect_V_nu(pi: Matrix2D, pi_0: Matrix2D, cost: Matrix2D, settings: Settings):
    dimX, dimY = settings.dimX, settings.dimY

    if settings.nu_0 is None:
        nu_0 = stationary_dist(pi_0.m)
    else:
        nu_0 = settings.nu_0
    a = (np.eye(pi.rows, dtype=int) - settings.gamma * pi.transpose().m)
    b = (1 - settings.gamma) * nu_0
    nu = np.linalg.solve(a, b)

    V = np.linalg.inv(
        np.eye(dimX * dimY) - settings.gamma * pi.m) @ cost.m.flatten()

    E = (V*nu).sum()
    return E

def sample_trajectory(pi: Matrix2D, nu_0, h, settings: Settings):
    dimX, dimY = settings.dimX, settings.dimY

    #Pi = pi.m.reshape((dimX, dimY, dimX, dimY))
    xy = dimY * (nu_0[0]) + nu_0[1]
    trajectory = [np.unravel_index(xy, (dimX, dimY))]
    for i in range(h):
        xy_prime = np.random.choice(np.arange(pi.rows), p=pi.m[xy])
        trajectory.append(np.unravel_index(xy_prime, (dimX, dimY)))
        xy = xy_prime

    return trajectory