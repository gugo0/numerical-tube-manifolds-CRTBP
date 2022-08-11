"""Functions needed in the simulation of the Circular Restricted Three Body Problem (CRTBP).
A module with all the functions necessary to compute the dynamics in the Poincaré section z = 0, dot(z) > 0.
Created on Sat Jun  25 12:36:20 2022

@author: Guglielmo Gomiero
"""

# imports
from numpy import ndarray
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import csv
import phase as ph


# Poincaré section
class section:
    """
    Find the points on the Poincaré section z=0, dot(z) > 0. Creates an event function that stops the integration
    started on initial conditions r=[x,y,z,px,py,pz] when the particle crosses the z=0 plane
    with upward velocity dot(z)>0.

    :param r: np.array
    :param m: float
    :return: np.array
    """
    def __init__(self,r,m):
        # Create event function, z=0
        def z_cross(t, r, m):
            return r[2]
        # Trigger event only when dot(z)>0
        z_cross.direction = 1
        z_cross.terminal = True

        # Initialize integration
        tspan = [0,100]
        sol = solve_ivp(ph.ham_eqs, tspan, r, args=(m,), atol=1e-12, rtol=1e-11, events=z_cross)

        # Store data from integration
        self.traj = sol.y
        self.energy = ph.energy(sol.y, m)


# Poincaré map
def map(s, E, m):
    """
    Defines the Poincaré map with energy E, for the section z=0 dot(z)>0, in a system with mass parameter m.
    :param s: np.array [x,y,px,py]
    :param E: float
    :param m: float
    :return: np.array [x,y,px,py]
    """
    # Extract coordinates
    x = s[0]
    y = s[1]
    px = s[2]
    py = s[3]
    # Calculate distances
    d1 = np.sqrt((x + m) ** 2 + y ** 2)
    d2 = np.sqrt((x + m - 1) ** 2 + y ** 2)
    # Calculate pz from these coordinates
    V = - (1 - m) / d1 - m / d2
    # debug
    # print('Map: ' + str(2*E - px**2 -py**2 + 2*x*py - 2*y*px - 2*V))
    # Since dot(z)=pz we only want positive pz to be coherent with the Poincaré section
    pz = np.sqrt(2*E - px**2 - py**2 + 2*x*py - 2*y*px - 2*V)

    sect = section([x, y, 1e-20, px, py, pz], m)

    # Extract only the needed coordinates
    s = np.zeros(4)
    s[0] = sect.traj[0, -1]
    s[1] = sect.traj[1, -1]
    s[2] = sect.traj[3, -1]
    s[3] = sect.traj[4, -1]

    return s


# Jacobian of Poincare map
def pJacob(s,E,m):
    """
    Calculates the Jacobian of the poincare map on point s

    Parameters
    ----------
    s :  np.array
        s = [x,y,px,py], point on the P map
    E : float
         Energy of P map
    m : float
        Mass parameter of system

    Returns
    -------
    Df :  np.array
         4x4 matrix, Jacobian of P map

    """
    # ! This term must not be too large (little accuracy) nor too small (high numerical error).
    # May need to be changed to better suit problem at hand
    h = 1e-7
    # Extract coordinates of s. Will be needed later
    x = s[0]
    y = s[1]
    px = s[2]
    py = s[3]

    # Numerical derivation
    # Sum h to each coordinate
    Px = map([x + h, y, px, py], E, m)
    Py = map([x, y + h, px, py], E, m)
    Ppx = map([x, y, px + h, py], E, m)
    Ppy = map([x, y, px, py + h], E, m)
    # Subtract h to each coordinate
    Mx = map([x - h, y, px, py], E, m)
    My = map([x, y - h, px, py], E, m)
    Mpx = map([x, y, px - h, py], E, m)
    Mpy = map([x, y, px, py - h], E, m)

    # Calculate derivatives matrix
    Df = np.zeros((4, 4))

    Df[0, 0] = (Px[0] - Mx[0]) / (2 * h)
    Df[0, 1] = (Py[0] - My[0]) / (2 * h)
    Df[0, 2] = (Ppx[0] - Mpx[0]) / (2 * h)
    Df[0, 3] = (Ppy[0] - Mpy[0]) / (2 * h)

    Df[1, 0] = (Px[1] - Mx[1]) / (2 * h)
    Df[1, 1] = (Py[1] - My[1]) / (2 * h)
    Df[1, 2] = (Ppx[1] - Mpx[1]) / (2 * h)
    Df[1, 3] = (Ppy[1] - Mpy[1]) / (2 * h)

    Df[2, 0] = (Px[2] - Mx[2]) / (2 * h)
    Df[2, 1] = (Py[2] - My[2]) / (2 * h)
    Df[2, 2] = (Ppx[2] - Mpx[2]) / (2 * h)
    Df[2, 3] = (Ppy[2] - Mpy[2]) / (2 * h)

    Df[3, 0] = (Px[3] - Mx[3]) / (2 * h)
    Df[3, 1] = (Py[3] - My[3]) / (2 * h)
    Df[3, 2] = (Ppx[3] - Mpx[3]) / (2 * h)
    Df[3, 3] = (Ppy[3] - Mpy[3]) / (2 * h)

    # Debug: verify that the map f is symplectic
    # J = np.zeros((4,4))
    # J[0,2] = 1
    # J[1,3] = 1
    # J[2,0] = -1
    # J[3,1] = -1
    # ver = np.transpose(Df) @ J @ Df
    # print(ver - J) # should be nearly 0
    return Df


# Root finding algorithm, Newton
class Newton:
    def __init__(self, s, E, m):
        """
        Finds t he fixed point of the P map, given a suitable starting condition s.
        Stores the iterations in Newton.s

        Parameters
        ----------
        s :  np.array
            . s = [x,y,px,py], point on the P map
        E : float
            . Energy of P map
        m : float
            . Mass parameter of system

        Returns
        -------
        None.

        """
        self.s = np.array([s])
        for i in range(25):
            # print('i= ' + str(i))
            f = map(s,E,m)
            # Define error function
            F = f - s
            print('Error= ' + str(F))

            Df = pJacob(s,E,m)
            # Define terms of newton iteration
            DF = Df - np.identity(4)
            invDF = np.linalg.inv(DF)

            # Debug
            # print(DF)
            # print(invDF)
            # print(np.linalg.det(DF))

            # Iterate Newton
            s_new = s - invDF@F
            
            # Store points found
            self.s = np.append(self.s, [s_new], axis=0)
            s = s_new
            print('step '+str(i)+' : ' + str(s_new))


# pz finder
def pz_finder(s, E, m):
    """
    Find the value of pz from s =[x,y,px,py], z=0 and the energy E.
    :param s: np.array
    :param E: float
    :param m: float
    :return: float
    """
    # Extract coordinates of s. Will be needed later
    x = s[0]
    y = s[1]
    px = s[2]
    py = s[3]
    d1 = np.sqrt((x + m) ** 2 + y ** 2)
    d2 = np.sqrt((x + m - 1) ** 2 + y ** 2)

    V = - (1 - m) / d1 - m / d2
    # We only want pz>0 to be coherent with the Poincare section
    pz = np.sqrt(2 * E - px ** 2 - py ** 2 + 2 * x * py - 2 * y * px - 2 * V)
    return pz


class Propagator:
    def __init__(self, opt_slast, opt_last, E_slast, E_last, E_target, m, file_name='results.csv'):
        """
        From 2 Lyapunov orbits fixed points propagates the family to a desired
        energy.
        The 2 orbits need to have very close energies: dE < 1e-4
        Stores the iterations in a .csv file

        Parameters
        ----------
        opt_slast :  np.array
             fixed point 0, s = [x,y,px,py]
        opt_last :  np.array
             fixed point 1, s = [x,y,px,py]
        E_slast :  float
             Energy for orbit 0
        E_last :  float
             Energy for orbit 1
        E_target :  float
             Desired energy
        m :  float
             mass parameter of system
        file_name : string, optional
            Name of .csv file. The default is 'results.csv'.

        Returns
        -------
        None.

        """
        dE = E_last - E_slast
        print('dE: ' + str(dE))
        # Save values to a csv table
        with open(file_name, 'w') as file:
            writ = csv.writer(file)
            writ.writerow([opt_slast[0], opt_slast[1], opt_slast[2], opt_slast[3], E_slast])

        with open(file_name, 'a') as file:
            writ = csv.writer(file)
            writ.writerow([opt_last[0], opt_last[1], opt_last[2], opt_last[3], E_last])
        # Store values as attributes for easy access
        self.results = np.array([opt_slast, opt_last])
        self.energy = np.array([E_slast, E_last])

        E_n = E_last
        n = 2

        while E_n < E_target:
            n += 1
            M = (1 / dE) * (opt_last - opt_slast)
            Q = opt_slast - E_slast * M

            E_n = E_last + dE
            guess_n = E_n * M + Q
            print(guess_n)

            newt = Newton(guess_n, E_n, m)
            print('Newton success')
            opt_n = newt.r[-1, :]
            # Print for debug
            print(n, opt_n, E_n)
            # Save as attributes
            self.results = np.append(self.results, [opt_n], axis=0)
            self.energy = np.append(self.energy, E_n)
            # Save to csv table
            with open(file_name, 'a') as file:
                writ = csv.writer(file)
                writ.writerow([opt_n[0], opt_n[1], opt_n[2], opt_n[3], E_n])

            opt_slast = opt_last.copy()
            opt_last = opt_n.copy()
            E_slast = E_last
            E_last = E_n