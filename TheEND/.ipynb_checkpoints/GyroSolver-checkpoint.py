import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objects as go
from IPython.display import HTML

from MvtODEs import*
from PhysicalQte import*

def Solve_Gyro(t1, t2, CI, params, solving_method='RK23', forcing='XY', nb_points=1000):
    '''
    Sam's work
    '''
    
    g, m, h, J1, J3, x0, p, f = params
    w = 2 * np.pi * f
    J1_ = J1 + m*(h**2)
    K = m * h * x0 * (w**2)
    p_psi = CI[-1] * J3
    t = np.linspace(t1, t2, nb_points, endpoint=True)
    ans = odeint(dSdt_ODEINT, y0=CI, t=t, args=(g, m, h, J1, J3, x0, p, w, J1_, K, p_psi, forcing))
    the_t = ans.T[0]
    phi_t = ans.T[2]
    psi_t = ans.T[4]
    the_t_d = ans.T[1]
    phi_t_d = ans.T[3]
    psi_t_d = ans.T[5]
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d


def Solve_Gyro_Fast(t1, t2, CI, params, solving_method='RK23', forcing='XY'):
    '''
    Sam's work
    '''
    
    g, m, h, J1, J3, x0, p, f = params
    w = 2 * np.pi * f
    J1_ = J1 + m * (h**2)
    K = m * h * x0 * (w**2)
    p_psi = (CI[-1] + CI[-3] * np.sin(CI[0])**2) * J3
    ans = solve_ivp(dSdt_IVP, t_span=(t1, t2), y0=CI, args=(g, m, h, J1, J3, x0, p, w, J1_, K, p_psi, forcing), method=solving_method)
    the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d = ans.y
    t = ans.t
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d


def Rabi_Chevron(exc_freq, tab_t_burst, params, CI, solving_method='RK23', forcing='XY'):
    '''
    Sam's work
    '''
    
    list_the = []
    list_t = []
    tab_theta = np.zeros( (len(exc_freq), len(tab_t_burst)) )
    
    for i in range(len(exc_freq)):
        
        params[-1] = exc_freq[i]
        
        for j in range(len(tab_t_burst)):
            tf = tab_t_burst[j]
            t, theta, _, _, _, _, _ = Solve_Gyro(0, tf, CI, params, solving_method=solving_method, forcing=forcing)
            #t, theta, _, _, _, _, _ = Solve_Gyro_Fast(0, tf, CI, params, solving_method=solving_method,   forcing=forcing)
            tab_theta[i, j] = theta[-1]
            list_the.append(theta)
            list_t.append(t)

            
    
    return tab_theta, list_the, list_t


def Cobra(dt1, dt2, dt3, CI, params, forcing='XY', list_solving_method=['RK23'] * 3, list_nb_points=[1000] * 3):
    '''Sam's work.'''
    
    T1 = dt1
    T2 = T1 +dt2
    T3 = T2 + dt3
    t1, the1, phi1, psi1, the_d1, phi_d1, psi_d1 = Solve_Gyro(0, T1, CI, params, solving_method=list_solving_method[0], forcing=forcing, nb_points=list_nb_points[0])
    CI1 = [the1[-1], the_d1[-1], phi1[-1], phi_d1[-1], psi1[-1], psi_d1[-1]]
    t2, the2, phi2, psi2, the_d2, phi_d2, psi_d2 = Solve_Gyro(T1, T2, CI1, params, solving_method=list_solving_method[1], forcing='FREE', nb_points=list_nb_points[1])
    CI2 = [the2[-1], the_d2[-1], phi2[-1], phi_d2[-1], psi2[-1], psi_d2[-1]]
    params2 = params
    params[-2] = 0
    t3, the3, phi3, psi3, the_d3, phi_d3, psi_d3 = Solve_Gyro(T2, T3, CI2, params2, solving_method=list_solving_method[2], forcing=forcing, nb_points=list_nb_points[2])

    t = np.concatenate([t1, t2, t3])
    the = np.concatenate([the1, the2, the3])
    phi = np.concatenate([phi1, phi2, phi3])
    psi = np.concatenate([psi1, psi2, psi3])
    the_d = np.concatenate([the_d1, the_d2, the_d3])
    phi_d = np.concatenate([phi_d1, phi_d2, phi_d3])
    psi_d = np.concatenate([psi_d1, psi_d2, psi_d3])

    return t, the, phi, psi, the_d, phi_d, psi_d
    

def Frequency_Sweep_MIN(exc_freq, params, CI, tf=80, solving_method='RK23', plot=True, forcing='XY'):
    '''Sam's work.'''
    
    tab_theta = []
    tab_theta_min = []
    tab_t = []
    
    for f in exc_freq:
        params[-1] = f
        t, theta, _, _, _, _, _ = Solve_Gyro(0, tf, CI, params, solving_method=solving_method, forcing=forcing)
        tab_theta_min.append( np.min(theta) * 180 / np.pi )
        tab_theta.append(theta)
        tab_t.append(t)

    tab_theta_min = np.array(tab_theta_min)
    f_max = exc_freq[tab_theta_min == np.min(tab_theta_min)]
    
    if plot:
        plt.figure()
        plt.scatter(exc_freq, tab_theta_min, marker='o')
        plt.xlabel(r"$f \; (\mathrm{Hz})$")
        plt.ylabel(r"$\theta_\mathrm{min} \; (\mathrm{deg})$")
        plt.title('Colatitude minimale en fonction du detuning')
        plt.show()
        
    return f_max, exc_freq, tab_theta, tab_theta_min, tab_t


def Frequency_Sweep_MAX(exc_freq, params, CI, tf=80, solving_method='RK23', plot=True, forcing='XY'):
    '''Sam's work.'''
    
    tab_theta = []
    tab_theta_max = []
    tab_t = []
    
    for f in exc_freq:
        params[-1] = f
        t, theta, _, _, _, _, _ = Solve_Gyro(0, tf, CI, params, solving_method=solving_method, forcing=forcing)
        tab_theta_max.append( np.max(theta) * 180 / np.pi )
        tab_theta.append(theta)
        tab_t.append(t)

    tab_theta_max = np.array(tab_theta_max)
    f_max = exc_freq[tab_theta_max == np.min(tab_theta_max)]
    
    if plot:
        plt.figure()
        plt.scatter(exc_freq, tab_theta_max, marker='o')
        plt.xlabel(r"$f \; (\mathrm{Hz})$")
        plt.ylabel(r"$\theta_\mathrm{max} \; (\mathrm{deg})$")
        plt.title('Colatitude maximale en fonction du detuning')
        plt.show()
    
    return f_max, exc_freq, tab_theta, tab_theta_max, tab_t


def Plot_Gyro_Angles(t, the, phi, psi):
    '''Sam's work.'''

    plt.figure(figsize=[16, 5])
    plt.subplot(1,3,1)
    plt.plot(t, the * 180 / np.pi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\theta \; (\text{deg})$')
    
    plt.subplot(1,3,2)
    plt.title(r"Angles d'Euler en fonction du temps")
    plt.plot(t, phi * 180 / np.pi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\phi \; (\text{deg})$')
    
    plt.subplot(1,3,3)
    plt.plot(t, psi * 180 / np.pi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\psi \; (\text{deg})$')
    plt.show()

    return None


def Plot_Gyro_Speed_Angles(t, the_d, phi_d, psi_d):
    '''Sam's work.'''

    plt.figure(figsize=[16, 5])
    plt.subplot(1,3,1)
    plt.plot(t, the_d * 180 / np.pi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\dot{\theta} \; (\text{deg})$')
    
    plt.subplot(1,3,2)
    plt.title(r"Dérivées temporelles des angles d'Euler en fonction du temps")
    plt.plot(t, phi_d * 180 / np.pi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\dot{\phi} \; (\text{deg})$')
    
    plt.subplot(1,3,3)
    plt.plot(t, psi_d * 180 / np.pi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\dot{\psi} \; (\text{deg})$')
    plt.show()

    return None


def Plot_Gyro_Momentum(t, p_the, p_phi, p_psi):
    '''Sam's work.'''

    plt.figure(figsize=[16, 5])
    plt.subplot(1,3,1)
    plt.plot(t, p_the)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\theta \; (\text{kg.m}^2)$')
    
    plt.subplot(1,3,2)
    plt.title(r"Moments conjugués en fonction du temps")
    plt.plot(t, p_phi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\phi \; (\text{kg.m}^2)$')
    
    plt.subplot(1,3,3)
    plt.plot(t, p_psi)
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\psi \; (\text{kg.m}^2)$')
    plt.show()

    return None
