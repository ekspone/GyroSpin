import numpy as np
import matplotlib.pyplot as plt
import diffeqpy
from diffeqpy import de, ode

from JuliaODE import*

def Gyro_Solver(tspan, CI, params, forcing='Free'):

    #the, the_d, phi, phi_d, psi, psi_d = u
    
    g, m, h, J1, J3, x0, Phi, f_exc = params

    Epp = m*g*h
    K = m*h*x0
    w = 2 * np.pi * f_exc
    J1_ = J1 + m*(h**2)

    p_psi = J3 * (CI[-1] + de.cos(CI[0]) * CI[-3])
    
    adapt_params = [Epp, J1_, p_psi, w, Phi, K]
    
    
    match forcing:
        case 'Free':
            #func_mvt = de.ODEFunction(Gyro_Mvt_Free, jac = Jacobian_Free)
            #prob = de.ODEProblem(func_mvt, CI, tspan, adapt_params)
            prob = de.ODEProblem(Gyro_Mvt_Free, CI, tspan, adapt_params)
        case 'XY':
            #func_mvt = de.ODEFunction(Gyro_Mvt_Forced_XY, jac = Jacobian_Forced_XY)
            #prob = de.ODEProblem(func_mvt, CI, tspan, adapt_params)
            prob = de.ODEProblem(Gyro_Mvt_Forced_XY, CI, tspan, adapt_params)
    

    fastprob = de.jit(prob)

    sol = de.solve(fastprob, de.AutoVern7(de.Rodas5()), abstol = 1e-5, reltol = 1e-5, maxiters=np.inf, alg_hints = ['stiff'])
    #sol = de.solve(fastprob#, de.Tsit5(), abstol = 1e-8, reltol = 1e-8, maxiters=np.inf)#, alg_hints = ['stiff'])

    sols_Julia = de.stack(sol.u)
    t = np.array(sol.t)
    the, the_d, phi, phi_d, psi, psi_d = np.array(sols_Julia)

    return t, the, the_d, phi, phi_d, psi, psi_d


def Rabi_Chevron(tab_exc_freq, tab_t_burst,  params, CI, forcing='XY'):
    
    list_the = []
    list_t = []
    tab_theta = np.zeros( (len(tab_exc_freq), len(tab_t_burst)) )

    for i in range(len(tab_exc_freq)):
        
        params[-1] = tab_exc_freq[i]
        
        for j in range(len(tab_t_burst)):
            tspan = (0, tab_t_burst[j])
            t, the, _, _, _, _, _ = Gyro_Solver(tspan, CI, params, forcing='XY')
            tab_theta[i, j] = the[-1]
            list_the.append(the)
            list_t.append(t)

            
    
    return tab_theta, list_the, list_t
    


def Cobra(dt1, dt2, dt3, CI, params, forcing='XY'):
    '''Sam's work.'''

    T1 = dt1
    T2 = T1 + dt2
    T3 = T2 + dt3
    tspan_1 = (0, T1)
    tspan_2 = (T1, T2)
    tspan_3 = (T2, T3)
    params[-2] = np.pi
    
    t1, the1, phi1, psi1, the_d1, phi_d1, psi_d1 = Gyro_Solver(tspan_1, CI, params, forcing=forcing)

    CI1 = [the1[-1], the_d1[-1], phi1[-1], phi_d1[-1], psi1[-1], psi_d1[-1]]
    t2, the2, phi2, psi2, the_d2, phi_d2, psi_d2 = Gyro_Solver(tspan_2, CI1, params, forcing='Free')

    CI2 = [the2[-1], the_d2[-1], phi2[-1], phi_d2[-1], psi2[-1], psi_d2[-1]]
    params2 = params
    params[-2] = 0
    t3, the3, phi3, psi3, the_d3, phi_d3, psi_d3 = Gyro_Solver(tspan_3, CI2, params2, forcing=forcing)

    t = np.concatenate([t1, t2, t3])
    the = np.concatenate([the1, the2, the3])
    phi = np.concatenate([phi1, phi2, phi3])
    psi = np.concatenate([psi1, psi2, psi3])
    the_d = np.concatenate([the_d1, the_d2, the_d3])
    phi_d = np.concatenate([phi_d1, phi_d2, phi_d3])
    psi_d = np.concatenate([psi_d1, psi_d2, psi_d3])

    return t, the, phi, psi, the_d, phi_d, psi_d
    









#-------------------------------------------------------------------------


def Plot_Gyro_Angles(t, the, phi, psi):
    '''Sam's work.'''

    plt.figure(figsize=[16, 5])
    plt.subplot(1,3,1)
    plt.grid()
    plt.plot(t, the * 180 / np.pi, 'b')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\theta \; (\text{deg})$')
    
    plt.subplot(1,3,2)
    plt.grid()
    plt.title(r"Angles d'Euler en fonction du temps")
    plt.plot(t, phi * 180 / np.pi, 'r')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\phi \; (\text{deg})$')
    
    plt.subplot(1,3,3)
    plt.grid()
    plt.plot(t, psi * 180 / np.pi, 'k')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\psi \; (\text{deg})$')
    plt.show()

    return None


def Plot_Gyro_Speed_Angles(t, the_d, phi_d, psi_d):
    '''Sam's work.'''

    plt.figure(figsize=[16, 5])
    plt.subplot(1,3,1)
    plt.grid()
    plt.plot(t, the_d * 180 / np.pi, 'b')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\dot{\theta} \; (\text{deg})$')
    
    plt.subplot(1,3,2)
    plt.grid()
    plt.title(r"Dérivées temporelles des angles d'Euler en fonction du temps")
    plt.plot(t, phi_d * 180 / np.pi, 'r')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\dot{\phi} \; (\text{deg})$')
    
    plt.subplot(1,3,3)
    plt.grid()
    plt.plot(t, psi_d * 180 / np.pi, 'k')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$\dot{\psi} \; (\text{deg})$')
    plt.show()

    return None


def Plot_Gyro_Momentum(t, p_the, p_phi, p_psi):
    '''Sam's work.'''

    plt.figure(figsize=[16, 5])
    plt.subplot(1,3,1)
    plt.grid()
    plt.plot(t, p_the, 'b')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\theta \; (\text{kg.m}^2)$')
    
    plt.subplot(1,3,2)
    plt.grid()
    plt.title(r"Moments conjugués en fonction du temps")
    plt.plot(t, p_phi, 'r')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\phi \; (\text{kg.m}^2)$')
    
    plt.subplot(1,3,3)
    plt.grid()
    plt.plot(t, p_psi, 'k')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\psi \; (\text{kg.m}^2)$')
    plt.show()

    return None


def Plot_Normalized_Gyro_Momentum(t, p_the, p_phi, p_psi):
    '''Sam's work.'''

    norm_p_the = p_the / np.max(p_the)
    norm_p_phi = np.abs(p_phi - np.mean(p_phi)) / np.mean(p_phi)
    norm_p_psi = np.abs(p_psi - np.mean(p_psi)) / np.mean(p_psi)
    
    plt.figure(figsize=[16, 5])
    plt.subplot(1,3,1)
    plt.grid()
    plt.plot(t, norm_p_the, 'b', label=r'$\frac{p_\theta}{\mathrm{max}(p_\theta)}$')
    plt.xlabel(r'$t \;(s)$')
    plt.legend()
    #plt.ylabel(r'$p_\theta \; (\text{kg.m}^2)$')
    
    plt.subplot(1,3,2)
    plt.grid()
    #plt.title(r"Moments conjugués en fonction du temps")
    plt.plot(t, norm_p_phi, 'r')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\phi \; (\text{kg.m}^2)$')
    
    plt.subplot(1,3,3)
    plt.grid()
    plt.plot(t, norm_p_psi, 'k')
    plt.xlabel(r'$t \;(s)$')
    plt.ylabel(r'$p_\psi \; (\text{kg.m}^2)$')
    plt.show()

    return None

