"""Functions needed in the simulation of the Circular Restricted Three Body Problem (CRTBP).
A module with all the functions necessary to compute the dynamics in a Hamiltonian flow.
Created on Thu Jun  23 20:56:20 2022

@author: Guglielmo Gomiero
"""

# imports
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import csv


# Hamilton equations
def ham_eqs(t, r, m):
    """Calculates the Hamilton equations of the CRTBP with mass parameter m.
    Return the time derivatives of the coordinates r = [x,y,z,px,py,pz]

    :param t: float time
    :param r: np.array [x,y,z,px,py,pz]
    :param m: float
    :return: np.array
    """
    # Extract coordinates
    x = r[0]
    y = r[1]
    z = r[2]
    px = r[3]
    py = r[4]
    pz = r[5]
    # Calculate distances from main bodies
    d1 = np.sqrt((x + m) ** 2 + y ** 2 + z ** 2)
    d2 = np.sqrt((x + m - 1) ** 2 + y ** 2 + z ** 2)
    # Calculate Hamilton equations (time derivatives)
    dx = px + y
    dy = py - x
    dz = pz
    dpx = py - ((1 - m) * (x + m)) / (d1 ** 3) - (m * (x + m - 1)) / (d2 ** 3)
    dpy = -px - ((1 - m) * y) / (d1 ** 3) - (m * y) / (d2 ** 3)
    dpz = - ((1 - m) * z) / (d1 ** 3) - (m * z) / (d2 ** 3)
    # Return derivatives as an array
    return np.array([dx, dy, dz, dpx, dpy, dpz])


# Energy value
def energy(r, m):
    """Calculates the energy for a given state vector r = [x,y,z,px,py,pz], in a
    system with mass parameter m.

    :param r: np.array
    :param m: float
    :return: float
    """
    # Extract coordinates
    x = r[0]
    y = r[1]
    z = r[2]
    px = r[3]
    py = r[4]
    pz = r[5]
    # Calculate distances form main bodies
    d1 = np.sqrt((x + m) ** 2 + y ** 2 + z ** 2)
    d2 = np.sqrt((x + m - 1) ** 2 + y ** 2 + z ** 2)
    # Calculate potential energy
    V = - (1 - m) / d1 - m / d2
    # Calculate mechanical energy
    E = .5 * (px ** 2 + py ** 2 + pz ** 2) - (x * py - y * px) + V
    return E


# Collinear Lagrange points
def lagrange_coll(m):
    """Calculates the collinear Lagrange points for a system with mass parameter m.
    Gives the x-coordinate of the three Lagrange points: root = [xL1,xL2,xL3]

    :param m: float
    :return: np.array
    """
    def func(x):
        return x - (1 - m) * (x + m) / np.abs((x + m)) ** 3 - m * (x + m - 1) / np.abs((x + m - 1)) ** 3

    x0 = np.array([1 - m - np.float_power(m / 3, 1 / 3),
                   1 - m + np.float_power(m / 3, 1 / 3),
                   -1 + m * 5 / 12])
    root = fsolve(func, x0)
    return root


# Jacobian
def jacobian(r, m):
    """Calculates the 6x6 Jacobian matrix for point r = [x,y,z,px,py,pz] in a system with mass
    parameter m.

    :param r: np.array
    :param m: float
    :return: np.array
    """
    # Extract coordinates, we only need x,y,z
    x = r[0]
    y = r[1]
    z = r[2]
    #px = r[3]
    #py = r[4]
    #pz = r[5]

    # Calculate distances
    d1 = np.sqrt((x + m) ** 2 + y ** 2 + z ** 2)
    d2 = np.sqrt((x + m - 1) ** 2 + y ** 2 + z ** 2)

    # Initialize Jacobian matrix
    J = np.zeros((6, 6))

    J[0, 1] = 1
    J[0, 3] = 1

    J[1, 0] = -1
    J[1, 4] = 1

    J[2, 5] = 1

    J[3, 0] = -(1 - m) * (1 / (d1 ** 3) - (3 * (x + m) ** 2) / (d1 ** 5)) - m * (
                1 / (d2 ** 3) - (3 * (x + m - 1) ** 2) / (d2 ** 5))
    J[3, 1] = (3 * (1 - m) * (x + m) * y) / (d1 ** 5) + (3 * m * (x + m - 1) * y) / (d2 ** 5)
    J[3, 2] = (3 * (1 - m) * (x + m) * z) / (d1 ** 5) + (3 * m * (x + m - 1) * z) / (d2 ** 5)
    J[3, 4] = 1

    J[4, 0] = J[3, 1]
    J[4, 1] = -(1 - m) * (1 / (d1 ** 3) - (3 * (y ** 2)) / (d1 ** 5)) - m * (1 / (d2 ** 3) - (3 * (y ** 2)) / (d2 ** 5))
    J[4, 2] = (3 * (1 - m) * y * z) / (d1 ** 5) + (3 * m * y * z) / (d2 ** 5)
    J[4, 3] = -1

    J[5, 0] = J[3, 2]
    J[5, 1] = J[4, 2]
    J[5, 2] = -(1 - m) * (1 / (d1 ** 3) - (3 * (z ** 2)) / (d1 ** 5)) - m * (1 / (d2 ** 3) - (3 * (z ** 2)) / (d2 ** 5))

    return J