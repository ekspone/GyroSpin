import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import HTML

def Gyro_Bloch(the, phi, psi):
    '''Sam's work. Unitary sphere, Sphère de Bloch'''

    x = np.sin(phi) * np.sin(the)
    y = -np.cos(phi) * np.sin(the)
    z = np.cos(the)

    return x, y, z


def Momentum_Weight(the, phi, psi, params):
    '''Moment associé au poids.'''
    g, m = params[0], params[1]
    momentum_the = m*g*np.sin(the)

    return momentum_the


def Momentum_Fe(t, the, phi, psi, params):
    '''Moment associé à la force centrifuge.'''
    m, x0, p, f = params[0], params[-3], params[-2], params[-1]
    w = 2*np.pi*f
    momentum_the = - m*x0*(w**2)*np.sin(w*t+p-phi)*np.cos(the)
    momentum_phi = m*x0*(w**2)*np.cos(w*t+p-phi)*np.sin(the)
    
    return momentum_the, momentum_phi


def Conjugated_Momentums(the, phi, psi, the_d, phi_d, psi_d, params):
    '''Moments conjugés.'''
    m, h, J1, J3 = params[1],  params[2], params[3],  params[4]
    J1_ = J1 + m*(h**2)
    p_psi = J3 * (psi_d + phi_d*np.cos(the))
    p_phi = J1_ * (np.sin(the)**2) * phi_d + np.cos(the)*p_psi
    p_the = J1_ * the_d
    return p_the, p_phi, p_psi


def p_psi_exp(params, CI):
    '''Calcule le P_psi initial.'''
    J3 = params[-4]
    p_psi0 = (CI[-1] + np.cos(CI[0]) * CI[-3]) * J3
    return p_psi0


def Get_Path(t, the, phi, psi, file_name="Gyro3D.png"):
    '''Génére la trajectoire du Gyro.'''

    x_t, y_t, z_t = Gyro_Bloch(the, phi, psi)
    
    i = 0
    f = len(t)

    layout = go.Layout(
        template="none",
        font=dict(family="Droid Serif"),
        scene=dict(
            xaxis_title=r"x",
            yaxis_title=r"y",
            zaxis_title=r"z",
            aspectratio=dict(x=1, y=1, z=1),
            camera_eye=dict(x=1.2, y=1.2, z=1.2),       
        ),
    )

    fig = go.Figure(layout=layout)
                    
    fig.add_scatter3d(x=[0], y=[0], z=[0], 
                      name="Contact point")
    
    fig.add_scatter3d(
        x=x_t[i:f],
        y=y_t[i:f],
        z=z_t[i:f],
        name="Path",
        mode="lines",
        line=dict(color=t, colorscale='Viridis', showscale=True),
    )

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01))

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
    fig.update_traces()
    fig.show()
    fig.write_image(file_name)



def Gyro_Carac_Values(params, CI):
    '''Calcule les fréquences et pulsations théoriques attendues'''
    p_psi0 = p_psi_exp(params, CI)
    g, m, h, J1, J3, x0, p, f = params
    omega_f = 2 * np.pi * f

    omega_L_th = m * g * h / (p_psi0)
    print(f'Larmor Pulsation (th) : {omega_L_th : >+20_.3f}')
    print(f'Larmor Frequency (th) : {omega_L_th/(2*np.pi) : >+20_.3f}')
    print(f'Larmor Period (th) : {2 * np.pi / omega_L_th : >+20_.3f} \n')
    
    omega_R_th = -0.5 * np.cos(p) * (m * h * x0 * (omega_f**2)) / p_psi0
    #omega_R_th = -0.5 * np.cos(params_f[-2]) * ( (m*h)**3 * g**2 * x0) / (p_psi_exp**3)
    
    print(f'Rabi Pulsation (th) : {omega_R_th : >+20_.3f}')
    print(f'Rabi Period (th) : {2 * np.pi / omega_R_th : >+20_.3f}')
    print(f'Temps de montée (th) : {np.pi / omega_R_th : >+20_.3f} \n')
    
    
    rapport_freq = float(omega_L_th / omega_R_th)
    print(f'Rapport des pulsations Larmor/Rabi : {rapport_freq : >+20_.3f} \n')
        
    print(f'Rapport Approx Gyroscopique : {0.5 * J3 * ((2 * np.pi * CI[-1])**2) /  (m*g*h) : >+20_.3f} \n')
    
    return None


def Hamiltonian_Terms(t, the, phi, psi, the_d, phi_d, psi_d, params, forcing='FREE'):
    '''Calcule les termes aparaissant dans l'Hamiltonien'''

    g, m, h, J1, J3, x0, p, f = params
    w = 2*np.pi*f
    J1_ = J1 + m*(h**2)
    K = m * h * x0 * (w**2)
    p_the, p_phi, p_psi = Conjugated_Momentums(the, phi, psi, the_d, phi_d, psi_d, params)
    
    Epp = m*g*h*np.cos(the)
    Nutation = 0.5 * J1_ * (np.sin(the)**2) * (phi_d)**2
    Ec_theta = (p_the**2) / (2 * J1_)
    Ec_psi = (p_psi**2) / (2 * J3)
    match forcing:
        case 'FREE':
            E_ext = 0
        case 'XY':
            E_ext = K*np.sin(the) * np.sin( w*t+p-phi )
        case 'X':
            E_ext = - K*np.sin(the) * sin(phi) * np.cos(w*t+p)
    return Ec_theta, Ec_psi, Nutation, Epp, E_ext



def Rabi_Ideal(t_burst, delta, f_R):
    tab_Rabi_th = np.zeros((len(delta), len(t_burst)))
    for i in range(len(delta)):
        for j in range(len(t_burst)):
            pulsation = 2 * np.pi * np.sqrt(delta[i]**2 + f_R**2) / 2
            num = (f_R**2) * np.sin( pulsation * t_burst )**2
            den = f_R**2 + delta[i]**2
            tab_Rabi_th[i, j] = num / den
    return tab_Rabi_th


def Rabi_Freq_Modified(delta, f_R, f_L):
    return f_R * (1 + delta / f_L)**2


def Rabi_Assym(t_burst, delta, f_R, f_L):
    tab_Rabi_th = np.zeros((len(delta), len(t_burst)))
    for i in range(len(delta)):
        for j in range(len(t_burst)):
            f_R_mod = Rabi_Freq_Modified(delta[i], f_R, f_L)
            pulsation = 2 * np.pi * np.sqrt(delta[i]**2 + f_R_mod**2) / 2
            num = (f_R_mod**2) * np.cos( pulsation * t_burst[j] )**2
            den = f_R_mod**2 + delta[i]**2
            tab_Rabi_th[i, j] = num / den
    return tab_Rabi_th

