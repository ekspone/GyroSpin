import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objects as go
from IPython.display import HTML


def the_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''
    the_dd_free = (m*g*h - phi_d * p_psi)*np.sin(the) / J1_ + 0.5 * np.sin(2*the)*(phi_d**2)
    return the_dd_free


def phi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''
    phi_dd_free = - (2*the_d*phi_d / np.tan(the)) + (p_psi*the_d)/(J1_ * np.sin(the))    
    return phi_dd_free


def psi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    #J1_ = J1 + m * (h**2)
    #K = (w**2) * m * x0 * h
    #
    psi_dd_free = (1 + np.cos(the)**2)*the_d*phi_d/np.sin(the) - p_psi*the_d/(J1_ * np.tan(the))
    #psi_dd_free = np.sin(the)*the_d*phi_d - np.cos(the) * phi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K)
    return psi_dd_free


#--------------------------------------------#--------------------------------------------
def the_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Working on...'''
    the_dd_free = the_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K)
    the_dd_forced = - K * np.sin(w*t+p - phi) * np.cos(the) / J1_
    the_dd = the_dd_free + the_dd_forced
    return the_dd

def phi_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''
    phi_dd_free = phi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K)
    phi_dd_forced = K * np.cos(w*t+p - phi) / (J1_ * np.sin(the))
    phi_dd = phi_dd_free + phi_dd_forced
    
    return phi_dd

def psi_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''
    psi_dd_free = psi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K)
    psi_dd_forced = - K * np.cos(w*t+p - phi) / (J1_ * np.tan(the))
    psi_dd = psi_dd_free + psi_dd_forced
    return psi_dd
#--------------------------------------------#--------------------------------------------


def the_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    the_dd_free = the_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K)
    the_dd_forced = K * np.cos(w*t+p) * np.sin(phi) * np.cos(the) / J1_
    the_dd = the_dd_free + the_dd_forced
    return the_dd


def phi_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    phi_dd_free = phi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K)
    phi_dd_forced = K * np.cos(w*t+p)*np.cos(phi) / (J1_ * np.sin(the))
    phi_dd = phi_dd_free + phi_dd_forced
    
    return phi_dd


def psi_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    psi_dd_free = psi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K)
    psi_dd_forced = - K * np.cos(w*t+p) * np.cos(phi) / (J1_ * np.tan(the))
    psi_dd = psi_dd_free + psi_dd_forced
    return psi_dd


def Identity(x):
    return x


def dSdt_IVP(t, S, g, m, h, J1, J3, x0, p, w, J1_, K, p_psi, forcing):
    the, z1, phi, z2, psi, z3 = S

    match forcing:
        case 'X':
            return np.array([
            Identity(z1),
            the_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z2),
            phi_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z3),
            psi_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            ])
        case 'XY':
            return np.array([
            Identity(z1),
            the_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z2),
            phi_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z3),
            psi_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            ])
        case 'FREE':
            return np.array([
            Identity(z1),
            the_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z2),
            phi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z3),
            psi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            ])


def dSdt_ODEINT(S, t, g, m, h, J1, J3, x0, p, w, J1_, K, p_psi, forcing):
    the, z1, phi, z2, psi, z3 = S
    match forcing:
        case 'X':
            return np.array([
            Identity(z1),
            the_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z2),
            phi_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z3),
            psi_dd_X(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            ])
        case 'XY':
            return np.array([
            Identity(z1),
            the_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z2),
            phi_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z3),
            psi_dd_XY(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            ])
        case 'FREE':
            return np.array([
            Identity(z1),
            the_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z2),
            phi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            Identity(z3),
            psi_dd_Free(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
            ])

