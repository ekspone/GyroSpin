import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objects as go
from IPython.display import HTML


t, h, g, m, x0, p, w, J1, J3 = smp.symbols(r't h g m x_0 \Phi \omega J_1, J_3', real=True)
the, phi, psi = smp.symbols(r'\theta \phi \psi', cls=smp.Function)
the = the(t)
phi = phi(t)
psi = psi(t)
# Derivatives
the_d = smp.diff(the,t)
phi_d = smp.diff(phi,t)
psi_d = smp.diff(psi,t)
# Second derivatives
the_dd = smp.diff(the_d,t)
phi_dd = smp.diff(phi_d,t)
psi_dd = smp.diff(psi_d,t)

R3 = smp.Matrix([[smp.cos(psi),-smp.sin(psi),0],
                 [smp.sin(psi),smp.cos(psi),0],
                 [0,0,1]])

R2 = smp.Matrix([[1,0,0],
                 [0,smp.cos(the),-smp.sin(the)],
                 [0,smp.sin(the),smp.cos(the)]])

R1 = smp.Matrix([[smp.cos(phi),-smp.sin(phi),0],
                 [smp.sin(phi),smp.cos(phi),0],
                 [0,0,1]])

R = R1*R2*R3

omega = smp.Matrix([phi_d*smp.sin(the)*smp.sin(psi)+the_d*smp.cos(psi),
                    phi_d*smp.sin(the)*smp.cos(psi)-the_d*smp.sin(psi),
                    phi_d*smp.cos(the)+psi_d])

I = smp.Matrix([[J1,0,0],[0,J1,0],[0,0,J3]])

xc = h * R @ smp.Matrix([0, 0, 1]) + smp.Matrix([x0 * smp.cos(w * t + p), 0, 0])
xc.simplify()
vc = smp.diff(xc, t)
vc.simplify()
vc_carre = vc.T.dot(vc)
vc_carre.simplify()

xc_free = h * R @ smp.Matrix([0, 0, 1])
xc_free.simplify()
vc_free = smp.diff(xc_free, t)
vc_free.simplify()
vc_carre_free = vc_free.T.dot(vc_free)
vc_carre_free.simplify()


# Kinetic, and potential energy
T_rot = (smp.Rational(1, 2) * omega.T.dot(I * omega).simplify())  # Rotational kinetic energy
T_c = smp.Rational(1, 2) * m * vc_carre  # Translational kinetic energy
T_c = T_c.simplify()
T = T_c + T_rot  # Total kinetic energy
V = m * g * h * smp.cos(the)  # Potential energy (gravitation)

T_free = smp.Rational(1, 2) * m * vc_carre_free + T_rot
T_free = T_free.simplify()

# Lagrangian
L = T_free - V
L = L.simplify()
L_f = T - V
L_f = L_f.simplify()

genCoordinates = [the, phi, psi]
genSpeeds = [the_d, phi_d, psi_d]
genAcceleration = [the_dd, phi_dd, psi_dd]

def EulerLagrange(lagrangian, genCoordinate, genSpeed, time):
    """
    Return the Euler-Lagrange Equation for the desired system of generalized coordinate

    ## Pamameters
    `lagrangian` : list or np.array
        Theta angle value across time 
    `phi` : list or np.array
        Phi angle value across time 
    `psi` : list or np.array
        Psi angle value across time 


    ## Return
    `R` : MutableDenseMatrix
        Change of base matrix

    """
    return smp.diff(lagrangian, genCoordinate) - smp.diff(smp.diff(lagrangian, genSpeed), time).simplify()

# FREE Euler-Lagrange equation
equations = [EulerLagrange(L, genCoordinates[i], genSpeeds[i], t) for i in range(3)]
sols = smp.solve(equations, (the_dd, phi_dd, psi_dd), simplify=False, rational=False)

dz1dt = smp.lambdify((t, g, h, m, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[the_dd])
dthedt = smp.lambdify(the_d, the_d)

dz2dt = smp.lambdify((t, g, h, m, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[phi_dd])
dphidt = smp.lambdify(phi_d, phi_d)

dz3dt = smp.lambdify((t, g, h, m, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[psi_dd])
dpsidt = smp.lambdify(psi_d, psi_d)



# FORCED Euler-Lagrange equation
equations_f = [EulerLagrange(L_f, genCoordinates[i], genSpeeds[i], t) for i in range(3)]
sols_f = smp.solve(equations_f, (the_dd, phi_dd, psi_dd), simplify=False, rational=False)

dz1dt_f = smp.lambdify((t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols_f[the_dd])
dthedt_f = smp.lambdify(the_d, the_d)

dz2dt_f = smp.lambdify((t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols_f[phi_dd])
dphidt_f = smp.lambdify(phi_d, phi_d)

dz3dt_f = smp.lambdify((t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols_f[psi_dd])
dpsidt_f = smp.lambdify(psi_d, psi_d)



#from numba import jit

#@jit(forceobj=True)
def dSdt_forced_IVP(t, S, g, m, h, J1, J3, x0, p, w):
    the, z1, phi, z2, psi, z3 = S
    return np.array([
        dthedt_f(z1),
        dz1dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
        dphidt_f(z2),
        dz2dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
        dpsidt_f(z3),
        dz3dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
    ])

def dSdt_forced_ODEINT(S, t, g, m, h, J1, J3, x0, p, w):
    the, z1, phi, z2, psi, z3 = S
    return np.array([
        dthedt_f(z1),
        dz1dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
        dphidt_f(z2),
        dz2dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
        dpsidt_f(z3),
        dz3dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
    ])

def dSdt_free_IVP(t, S, g, m, h, J1, J3):
    the, z1, phi, z2, psi, z3 = S
    return np.array([
        dthedt(z1),
        dz1dt(t, g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
        dphidt(z2),
        dz2dt(t, g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
        dpsidt(z3),
        dz3dt(t, g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
    ])

def dSdt_free_ODEINT(S, t, g, m, h, J1, J3):
    the, z1, phi, z2, psi, z3 = S
    return np.array([
        dthedt(z1),
        dz1dt(t, g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
        dphidt(z2),
        dz2dt(t, g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
        dpsidt(z3),
        dz3dt(t, g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
    ])

def Solve_Gyro_Free(t1, t2, CI, params, solving_method='RK23'):
    '''
    Sam's work
    '''
    
    g, m, h, J1, J3, x0, p, w = params
    t = np.linspace(t1, t2, 10000, endpoint=True)
    ans = odeint(dSdt_free_ODEINT, y0=CI, t=t, args=(g, m, h, J1, J3))
    the_t = ans.T[0]
    phi_t = ans.T[2]
    psi_t = ans.T[4]
    the_t_d = ans.T[1]
    phi_t_d = ans.T[3]
    psi_t_d = ans.T[5]
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d


def Solve_Gyro_Free_Fast(t1, t2, CI, params, solving_method='RK23'):
    '''
    Sam's work
    '''
    
    g, m, h, J1, J3, x0, p, w = params
    ans = solve_ivp(dSdt_free_IVP, t_span=(t1, t2), y0=CI, args=(g, m, h, J1, J3), method=solving_method)
    the_t, phi_t, phi_t, the_t_d, phi_t_d, psi_t_d = ans.y
    t = ans.t
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d


def Solve_Gyro_Forced(t1, t2, CI, params, solving_method='RK23'):
    '''
    Sam's work
    '''
    g, m, h, J1, J3, x0, p, w = params
    w = 2 * np.pi * w
    t = np.linspace(t1, t2, 10000, endpoint=True)
    ans = odeint(dSdt_forced_ODEINT, y0=CI, t=t, args=(g, m, h, J1, J3, x0, p, w))
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
    ans = solve_ivp(dSdt_forced_IVP, t_span=(t1, t2), y0=CI, args=(g, m, h, J1, J3, x0, p, w), method=solving_method)
    the_t, phi_t, phi_t, the_t_d, phi_t_d, psi_t_d = ans.y
    t = ans.t
    
    return t, the_t, phi_t, psi_t, the_t_d, phi_t_d, psi_t_d


def Pulse_Pi_Demi(t1, t2, CI, params, solving_method='RK23', fast=False):
    '''
    Sam's work.
    '''
    if fast:
        time1, the_t1, phi_t1, psi_t1, the_t1_d, phi_t1_d, psi_t1_d = Solve_Gyro_Forced_Fast(0, t1, CI, params, solving_method)
        CI2 = [the_t1[-1], the_t1_d[-1], phi_t1[-1], phi_t1_d[-1], psi_t1[-1], psi_t1_d[-1]]
        time2, the_t2, phi_t2, psi_t2, the_t2_d, phi_t2_d, psi_t2_d = Solve_Gyro_Free(0, t2, CI2, params, solving_method)
    else:
        time1, the_t1, phi_t1, psi_t1, the_t1_d, phi_t1_d, psi_t1_d = Solve_Gyro_Forced(0, t1, CI, params, solving_method)
        CI2 = [the_t1[-1], the_t1_d[-1], phi_t1[-1], phi_t1_d[-1], psi_t1[-1], psi_t1_d[-1]]
        time2, the_t2, phi_t2, psi_t2, the_t2_d, phi_t2_d, psi_t2_d = Solve_Gyro_Free(0, t2, CI2, params, solving_method)

    T = np.concatenate((time1, time2 + t1))
    THE = np.concatenate((the_t1, the_t2))
    PHI = np.concatenate((phi_t1, phi_t2))
    PSI = np.concatenate((psi_t1, psi_t2))
    THE_d = np.concatenate((the_t1_d, the_t2_d))
    PHI_d = np.concatenate((phi_t1_d, phi_t2_d))
    PSI_d = np.concatenate((psi_t1_d, psi_t2_d))

    return T, THE, PHI, PSI


def Cobra(t1, t2, t3, t4, CI, params_data, p1, p2, solving_method='RK23', fast=True):
    '''
    Sam's work.
    '''
    params = params_data
    params[-2] = p1
    time1, the1, phi1, psi1, the1_d, phi1_d, psi1_d = Pulse_Pi_Demi(t1, t2, CI, params, solving_method, fast)
    CI2 = [the1[-1], the1_d[-1], phi1[-1], phi1_d[-1], psi1[-1], psi1_d[-1]]
    params[-2] = p2
    time2, the2, phi2, psi2, the2_d, phi2_d, psi2_d = Pulse_Pi_Demi(t3, t4, CI2, params, solving_method, fast) 
    T = np.concatenate((time1, time2 + t3))
    THE = np.concatenate((the1, the2))
    PHI = np.concatenate((phi1, phi2))
    PSI = np.concatenate((psi1, psi2))
    THE_d = np.concatenate((the1_d, the2_d))
    PHI_d = np.concatenate((phi1_d, phi2_d))
    PSI_d = np.concatenate((psi1_d, psi2_d))

    return T, THE, PHI, PSI


def Frequency_array(f1, f2, step):
    return np.arange(f1, f2 + step, step)

def Amplitude_array(x1, x2, step):
    return np.arange(x1, x2 + step, step)


def Frequency_Sweep(f1, f2, step_f, params, tf=80 , CI = [np.pi / 12, 0, 0, 0, 0, 2*np.pi*200], solving_method='RK23', plot=True):
    exc_freq = Frequency_array(f1, f2, step_f)
    tab_theta = []
    tab_theta_max = []
    tab_t = []
    
    for f in exc_freq:
        params[-1] = f
        t, theta, _, _, _, _, _ = Solve_Gyro_Forced_Fast(0, tf, CI, params, solving_method=solving_method)
        tab_theta_max.append( np.max(theta) * 180 / np.pi )
        tab_theta.append(theta)
        tab_t.append(t)
    
    f_max = exc_freq[tab_theta_max == np.max(tab_theta_max)]
    tab_theta_max = np.array(tab_theta_max)

    if plot:
        plt.figure()
        plt.scatter(exc_freq, tab_theta_max, marker='o')
        plt.xlabel(r"$f \; (\mathrm{Hz})$")
        plt.ylabel(r"$\theta_\mathrm{max} \; (\mathrm{deg})$")
    
    return f_max, exc_freq, tab_theta, tab_theta_max, tab_t


def Amplitude_Sweep(x1, x2, step_x, params, tf=350 , CI = [np.pi / 24, 0, 0, 0, 0, 2*np.pi*200], solving_method='RK23', plot=True):
    exc_x = Amplitude_array(x1, x2, step_x) * 1e-2
    tab_theta = []
    tab_t = []
    dt = []
    for x in exc_x:
        params[-3] = x
        t, theta, _, _, _, _, _ = Solve_Gyro_Forced_Fast(0, tf, CI, params, solving_method=solving_method)
        tab_theta.append(theta)
        tab_t.append(t)
        dt.append( t[theta == np.max(theta)][0] )

        if plot:
            plt.figure()
            plt.plot(t, theta)

            
    
    return tab_t, tab_theta, exc_x * 1e2, dt
    

def Double_Sweep(f1, f2, step_f, x1, x2, step_x, params, tf=100, CI = [np.pi / 12, 0, 0, 0, 0, 2*np.pi*200], solving_method='RK23', plot=True):
    
    exc_freq = Frequency_array(f1, f2, step_f)
    tab_x = Amplitude_array(x1, x2, step_x) * 1e-2
    tab_theta = np.zeros((len(tab_x), len(exc_freq)), dtype='object')
    tab_theta_max = np.zeros((len(tab_x), len(exc_freq)))
    tab_t = np.zeros((len(tab_x), len(exc_freq)), dtype='object')
    f_max = []
    
    for i in range(len(tab_x)):
        
        params[-3] = tab_x[i]
        
        for j in range(len(exc_freq)):
            params[-1] = exc_freq[j]
            t, theta, _, _, _, _, _ = Solve_Gyro_Forced_Fast(0, tf, CI, params, solving_method=solving_method)
            tab_theta_max[i, j] = np.max(theta * 180 / np.pi)
            tab_theta[i, j] = theta
            tab_t[i, j] = t
            
        f_max.append(exc_freq[tab_theta_max[i, :] == np.max(tab_theta_max[i, :])])
            
    
    return f_max, exc_freq, tab_x, tab_theta, tab_theta_max, tab_t





def Plot_Gyro_Path(the, phi):
    '''
    Sam's work
    '''
    x = np.sin(phi) * np.sin(the)
    y = -np.cos(phi) * np.sin(the)
    z = np.cos(the)

    i = 0
    f = len(x)

    layout = go.Layout(
        title=r"Plot Title",
        scene=dict(
            xaxis_title=r"x",
            yaxis_title=r"y",
            zaxis_title=r"z",
            aspectratio=dict(x=1, y=1, z=1),
            camera_eye=dict(x=1.2, y=1.2, z=1.2),
        ),
    )

    fig = go.Figure(layout=layout)
    fig.add_scatter3d(x=[0], y=[0], z=[0])
    fig.add_scatter3d(
        x=x[i:f], y=y[i:f], z=z[i:f], mode="lines", line=dict(color="green", width=2)
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-1, 1],
            ),
            yaxis=dict(
                range=[-1, 1],
            ),
            zaxis=dict(
                range=[-1, 1],
            ),
        )
    )

    path = HTML(fig.to_html(default_width=1000, default_height=600))
    return path



    


