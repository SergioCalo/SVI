import unittest
import numpy as np
from sinkhorn.sinkhorn_PI import Matrix2D, Settings, SPI


class TestSVI(unittest.TestCase):
    def test_2x2_matrices(self):
        settings = Settings(eta=1, gamma=0.95, N=1, K=100, epsilon=None, dimX=2, dimY=2, round=False, eta_decay=False, nu_0=None)
        Px = Matrix2D(np.array([[0.5000, 0.5000],
                                [0.2500, 0.7500]]), 2, 2)
        Py = Matrix2D(np.array([[0.7500, 0.2500],
                                [0.5000, 0.5000]]), 2, 2)
        cost = Matrix2D(np.array([[1, 0], [0, 1]]), 2, 2)
        pi = SPI(Px=Px, Py=Py, cost=cost, settings=settings)
        pi_compare = np.array([[0. ,  0.25, 0.75, 0.  ],
                     [0. ,  0.5 , 0.5,  0.  ],
                     [0.  , 0.25, 0.75, 0.  ],
                     [0.  , 0.5 , 0.5 , 0.  ]])

        self.assertAlmostEqual(sum(sum(pi.m-pi_compare)), 0.0, delta=1.0e-02)

    def test_2x3_matrices(self):
        settings = Settings(eta=1, gamma=0.95, N=1, K=100, epsilon=None, dimX=3, dimY=2, round=False, eta_decay=False, nu_0=None)
        Px = Matrix2D(np.array([[0.5000, 0.25000, 0.25],
                                [0.2500, 0.500, 0.25],
                                [0.25, 0.25, 0.5]]), 3, 3)
        Py = Matrix2D(np.array([[0.7500, 0.2500],
                                [0.5000, 0.5000]]), 2, 2)
        cost = Matrix2D(np.array([[1, 0, 1], [1, 0, 1]]), 2, 2)
        pi = SPI(Px=Px, Py=Py, cost=cost, settings=settings)
        pi_compare = np.array([[0.  ,  0.25 , 0.17,  0.   , 0.58 , 0.   ],
                                 [0.  ,  0.5 ,  0.113, 0.   , 0.387 ,0.   ,],
                                 [0. ,   0.25 , 0.277, 0.  ,  0.473, 0.   ],
                                 [0.   , 0.371 ,0.098 ,0.129, 0.402, 0.   ],
                                 [0.  ,  0.25 , 0.096, 0.  ,  0.654, 0.   ],
                                 [0.,    0.385 ,0. ,   0.115, 0.5 ,  0.   ]]
                                )

        self.assertAlmostEqual(sum(sum(pi.m-pi_compare)), 0.0, delta=1.0e-02)


if __name__ == '__main__':
    unittest.main()
