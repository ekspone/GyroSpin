import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objects as go
from IPython.display import HTML


def the_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    the_dd_free = (m*g*h - phi_d * p_psi)*np.sin(the) / J1_ + 0.5 * np.sin(2*the)*(phi_d**2)
    the_dd_forced = K * np.cos(w*t+p) * np.sin(phi) * np.cos(the) / J1_
    the_dd = the_dd_free + the_dd_forced
    return the_dd


def phi_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    phi_dd_free = - (2*the_d*phi_d / np.tan(the)) + (p_psi*the_d)/(J1_ * np.sin(the))
    phi_dd_forced = K * np.cos(w*t+p)*np.cos(phi) / (J1_ * np.sin(the))
    phi_dd = phi_dd_free + phi_dd_forced
    
    return phi_dd


def psi_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    J1_ = J1 + m * (h**2)
    K = (w**2) * m * x0 * h
    psi_dd_free = (1 + np.cos(the)**2)*the_d*phi_d/np.sin(the) - p_psi*the_d/(J1_ * np.tan(the))
    psi_dd_forced = - K * np.cos(w*t+p) * np.cos(phi) / (J1_ * np.tan(the))
    psi_dd = psi_dd_free + psi_dd_forced
    return psi_dd


def Identity(x):
    return x


def dSdt_forced_IVP(t, S, g, m, h, J1, J3, x0, p, w, J1_, K, p_psi):
    the, z1, phi, z2, psi, z3 = S
    return np.array([
        Identity(z1),
        the_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
        Identity(z2),
        phi_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
        Identity(z3),
        psi_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
    ])


def dSdt_forced_ODEINT(S, t, g, m, h, J1, J3, x0, p, w, J1_, K, p_psi):
    the, z1, phi, z2, psi, z3 = S
    return np.array([
        Identity(z1),
        the_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
        Identity(z2),
        phi_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
        Identity(z3),
        psi_dd(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
    ])


def Solve_Gyro_Forced(t1, t2, CI, params, solving_method='RK23'):
    '''
    Sam's work
    '''
    g, m, h, J1, J3, x0, p, w = params
    w = 2 * np.pi * w
    J1_ = J1 + m*(h**2)
    K = m * h * x0 * (w**2)
    p_psi = CI[-1] * J3
    t = np.linspace(t1, t2, 10000, endpoint=True)
    ans = odeint(dSdt_forced_ODEINT, y0=CI, t=t, args=(g, m, h, J1, J3, x0, p, w, J1_, K, p_psi))
    the_t = ans.T[0]
    phi_t = ans.T[2]
    psi_t = ans.T[4]
    the_t_d = ans.T[1]
    phi_t_d = ans.T[3]
    psi_t_d = ans.T[5]
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d


def Solve_Gyro_Forced_Fast(t1, t2, CI, params, solving_method='RK23'):
    '''
    Sam's work
    '''
    g, m, h, J1, J3, x0, p, w = params
    
    w = 2 * np.pi * w
    J1_ = J1 + m * (h**2)
    K = m * h * x0 * (w**2)
    p_psi = (CI[-1] + CI[-3] * np.sin(CI[0])**2) * J3
    ans = solve_ivp(dSdt_forced_IVP, t_span=(t1, t2), y0=CI, args=(g, m, h, J1, J3, x0, p, w, J1_, K, p_psi), method=solving_method)
    the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d = ans.y
    t = ans.t
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d


def Rabi_Chevron(exc_freq, tab_t_burst, params, CI, solving_method='RK23'):

    list_the = []
    list_t = []
    tab_theta = np.zeros( (len(exc_freq), len(tab_t_burst)) )
    
    for i in range(len(exc_freq)):
        
        params[-1] = exc_freq[i]
        
        for j in range(len(tab_t_burst)):
            tf = tab_t_burst[j]
            t, theta, _, _, _, _, _ = Solve_Gyro_Forced_Fast(0, tf, CI, params, solving_method=solving_method)
            tab_theta[i, j] = theta[-1]
            list_the.append(theta)
            list_t.append(t)

            
    
    return tab_theta, list_the, list_t

def Frequency_Sweep_MIN(exc_freq, params, tf=80 , CI = [np.pi / 12, 0, 0, 0, 0, 2*np.pi*200], solving_method='RK23', plot=True):
    tab_theta = []
    tab_theta_min = []
    tab_t = []
    
    for f in exc_freq:
        params[-1] = f
        t, theta, _, _, _, _, _ = Solve_Gyro_Forced_Fast(0, tf, CI, params, solving_method=solving_method)
        tab_theta_min.append( np.min(theta) * 180 / np.pi )
        tab_theta.append(theta)
        tab_t.append(t)
    
    f_max = exc_freq[tab_theta_min == np.min(tab_theta_min)]
    tab_theta_min = np.array(tab_theta_min)

    if plot:
        plt.figure()
        plt.scatter(exc_freq, tab_theta_min, marker='o')
        plt.xlabel(r"$f \; (\mathrm{Hz})$")
        plt.ylabel(r"$\theta_\mathrm{max} \; (\mathrm{deg})$")
    
    return f_max, exc_freq, tab_theta, tab_theta_min, tab_t



def the_dd_SUM(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    the_dd_free = (m*g*h - phi_d * p_psi)*np.sin(the) / J1_ + 0.5 * np.sin(2*the)*(phi_d**2)
    the_dd_forced = K * (np.sin(w*t+p + phi) - np.sin(w*t + p - phi)) * np.cos(the) / (2*J1_)
    the_dd = the_dd_free + the_dd_forced
    return the_dd


def phi_dd_SUM(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    phi_dd_free = - (2*the_d*phi_d / np.tan(the)) + (p_psi*the_d)/(J1_ * np.sin(the))
    phi_dd_forced = K * (np.cos(w*t+p + phi) + np.cos(w*t + p - phi)) / (2*J1_ * np.sin(the))
    phi_dd = phi_dd_free + phi_dd_forced
    
    return phi_dd


def psi_dd_SUM(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, K):
    '''Docstring to do.'''

    J1_ = J1 + m * (h**2)
    K = (w**2) * m * x0 * h
    psi_dd_free = (1 + np.cos(the)**2)*the_d*phi_d/np.sin(the) - p_psi*the_d/(J1_ * np.tan(the))
    psi_dd_forced = - K * (np.cos(w*t+p + phi) + np.cos(w*t + p - phi)) / (2*J1_ * np.tan(the))
    psi_dd = psi_dd_free + psi_dd_forced
    return psi_dd

def dSdt_forced_IVP_SUM(t, S, g, m, h, J1, J3, x0, p, w, J1_, K, p_psi):
    the, z1, phi, z2, psi, z3 = S
    return np.array([
        Identity(z1),
        the_dd_SUM(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
        Identity(z2),
        phi_dd_SUM(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
        Identity(z3),
        psi_dd_SUM(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3, p_psi, J1_, K),
    ])


def Solve_Gyro_Forced_Fast_SUM(t1, t2, CI, params, solving_method='RK23'):
    '''
    Sam's work
    '''
    g, m, h, J1, J3, x0, p, w = params
    
    w = 2 * np.pi * w
    J1_ = J1 + m * (h**2)
    K = m * h * x0 * (w**2)
    p_psi = (CI[-1] + CI[-3] * np.sin(CI[0])**2) * J3
    ans = solve_ivp(dSdt_forced_IVP_SUM, t_span=(t1, t2), y0=CI, args=(g, m, h, J1, J3, x0, p, w, J1_, K, p_psi), method=solving_method)
    the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d = ans.y
    t = ans.t
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d

def Frequency_Sweep_MIN_SUM(exc_freq, params, tf=80 , CI = [np.pi / 12, 0, 0, 0, 0, 2*np.pi*200], solving_method='RK23', plot=True):
    tab_theta = []
    tab_theta_min = []
    tab_t = []
    
    for f in exc_freq:
        params[-1] = f
        t, theta, _, _, _, _, _ = Solve_Gyro_Forced_Fast_SUM(0, tf, CI, params, solving_method=solving_method)
        tab_theta_min.append( np.min(theta) * 180 / np.pi )
        tab_theta.append(theta)
        tab_t.append(t)
    
    f_max = exc_freq[tab_theta_min == np.min(tab_theta_min)]
    tab_theta_min = np.array(tab_theta_min)

    if plot:
        plt.figure()
        plt.scatter(exc_freq, tab_theta_min, marker='o')
        plt.xlabel(r"$f \; (\mathrm{Hz})$")
        plt.ylabel(r"$\theta_\mathrm{max} \; (\mathrm{deg})$")
    
    return f_max, exc_freq, tab_theta, tab_theta_min, tab_t