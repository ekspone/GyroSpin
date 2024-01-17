import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import odeint
import plotly.graph_objects as go
from IPython.display import HTML
from scipy.optimize import curve_fit
from wotan import flatten


def convert_angle(angle):
    return np.mod(angle, 2 * np.pi) * 180 / np.pi


def lin(x, a, b):
    return a * x + b


def Get_frequency(t, angle, freq0, angle0):
    tab, pcov = curve_fit(lin, t, angle, p0=[freq0, angle0])
    freq = tab[0] / (2 * np.pi)
    phi0 = tab[1]
    return freq, phi0


def Solve_Gyro_Forced_XY(time, CI, params, omega_excitation, plot=True):
    t, h, g, m, x0, w = smp.symbols("t h g m x_0, w", real=True)
    the, phi, psi = smp.symbols(r"\theta \phi \psi", cls=smp.Function)
    the = the(t)
    phi = phi(t)
    psi = psi(t)
    # Derivatives
    the_d = smp.diff(the, t)
    phi_d = smp.diff(phi, t)
    psi_d = smp.diff(psi, t)
    # Second derivatives
    the_dd = smp.diff(the_d, t)
    phi_dd = smp.diff(phi_d, t)
    psi_dd = smp.diff(psi_d, t)
    R3 = smp.Matrix(
        [[smp.cos(psi), -smp.sin(psi), 0], [smp.sin(psi), smp.cos(psi), 0], [0, 0, 1]]
    )

    R2 = smp.Matrix(
        [[1, 0, 0], [0, smp.cos(the), -smp.sin(the)], [0, smp.sin(the), smp.cos(the)]]
    )

    R1 = smp.Matrix(
        [[smp.cos(phi), -smp.sin(phi), 0], [smp.sin(phi), smp.cos(phi), 0], [0, 0, 1]]
    )

    R = R1 * R2 * R3
    xc = h * R @ smp.Matrix([0, 0, 1]) + smp.Matrix(
        [x0 * smp.cos(w * t), x0 * smp.sin(w * t), 0]
    )
    vc = smp.diff(xc, t)
    vc_carre = vc.T.dot(vc).simplify()
    omega = smp.Matrix(
        [
            phi_d * smp.sin(the) * smp.sin(psi) + the_d * smp.cos(psi),
            phi_d * smp.sin(the) * smp.cos(psi) - the_d * smp.sin(psi),
            phi_d * smp.cos(the) + psi_d,
        ]
    )
    J1, J3 = smp.symbols("J_1, J_3", real=True)
    I = smp.Matrix([[J1, 0, 0], [0, J1, 0], [0, 0, J3]])
    T_rot = smp.Rational(1, 2) * omega.T.dot(I * omega).simplify()
    T_c = smp.Rational(1, 2) * m * vc_carre

    T = T_c + T_rot

    V = m * g * h * smp.cos(the)
    L = T - V
    L = L.simplify()

    LE1 = smp.diff(L, the) - smp.diff(smp.diff(L, the_d), t)
    LE1 = LE1.simplify()

    LE2 = smp.diff(L, phi) - smp.diff(smp.diff(L, phi_d), t)
    LE2 = LE2.simplify()

    LE3 = smp.diff(L, psi) - smp.diff(smp.diff(L, psi_d), t)
    LE3 = LE3.simplify()
    sols = smp.solve(
        [LE1, LE2, LE3], (the_dd, phi_dd, psi_dd), simplify=False, rational=False
    )
    dz1dt_f = smp.lambdify(
        (t, g, h, m, x0, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[the_dd]
    )
    dthedt_f = smp.lambdify(the_d, the_d)

    dz2dt_f = smp.lambdify(
        (t, g, h, m, x0, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[phi_dd]
    )
    dphidt_f = smp.lambdify(phi_d, phi_d)

    dz3dt_f = smp.lambdify(
        (t, g, h, m, x0, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[psi_dd]
    )
    dpsidt_f = smp.lambdify(psi_d, psi_d)

    def dSdt(S, t):
        the, z1, phi, z2, psi, z3 = S
        return [
            dthedt_f(z1),
            dz1dt_f(t, g, h, m, x0, w, J1, J3, the, phi, psi, z1, z2, z3),
            dphidt_f(z2),
            dz2dt_f(t, g, h, m, x0, w, J1, J3, the, phi, psi, z1, z2, z3),
            dpsidt_f(z3),
            dz3dt_f(t, g, h, m, x0, w, J1, J3, the, phi, psi, z1, z2, z3),
        ]

    g, m, h, J1, J3, x0 = params  ######### IMPORTANT

    w = omega_excitation

    t = time
    # Initial Condition 1
    ans = odeint(dSdt, y0=CI, t=t)

    path = 0

    the_t = ans.T[0]
    phi_t = ans.T[2]
    psi_t = ans.T[4]
    x_t = np.sin(phi_t) * np.sin(the_t)
    y_t = -np.cos(phi_t) * np.sin(the_t)
    z_t = np.cos(the_t)

    freq_precession, phi0 = Get_frequency(t, phi_t, 1, 0)
    freq_rotation, psi0 = Get_frequency(t, psi_t, CI[-1], 0)

    if plot:
        print("precession frequency =", freq_precession)
        print("rotation frequency =", freq_rotation)
        print(freq_rotation / freq_precession)
        print((J3 * (2 * np.pi * freq_rotation) ** 2) / (m * g * h))
        plt.figure(figsize=[15, 5])
        plt.subplot(1, 2, 1)
        plt.plot(t, the_t * 180 / np.pi)
        plt.ylabel(r"$\theta(t) \; (\mathrm{degrees})$")
        plt.xlabel(r"$t \; (\mathrm{s})$")

        plt.subplot(1, 2, 2)
        plt.plot(t, convert_angle(phi_t))
        plt.ylabel(r"$\phi(t) \; (\mathrm{degrees})$")
        plt.xlabel(r"$t \; (\mathrm{s})$")
        plt.show()

        i = 0
        f = 1000000

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
            x=x_t[i:f],
            y=y_t[i:f],
            z=z_t[i:f],
            mode="lines",
            line=dict(color="green", width=2),
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

    return the_t, phi_t, psi_t, x_t, y_t, z_t, path


def Solve_Gyro_Free(time, CI, params, plot=True):
    t, h, g, m = smp.symbols("t h g m", real=True)
    the, phi, psi = smp.symbols(r"\theta \phi \psi", cls=smp.Function)
    the = the(t)
    phi = phi(t)
    psi = psi(t)
    # Derivatives
    the_d = smp.diff(the, t)
    phi_d = smp.diff(phi, t)
    psi_d = smp.diff(psi, t)
    # Second derivatives
    the_dd = smp.diff(the_d, t)
    phi_dd = smp.diff(phi_d, t)
    psi_dd = smp.diff(psi_d, t)
    R3 = smp.Matrix(
        [[smp.cos(psi), -smp.sin(psi), 0], [smp.sin(psi), smp.cos(psi), 0], [0, 0, 1]]
    )

    R2 = smp.Matrix(
        [[1, 0, 0], [0, smp.cos(the), -smp.sin(the)], [0, smp.sin(the), smp.cos(the)]]
    )

    R1 = smp.Matrix(
        [[smp.cos(phi), -smp.sin(phi), 0], [smp.sin(phi), smp.cos(phi), 0], [0, 0, 1]]
    )

    R = R1 * R2 * R3
    xc = h * R @ smp.Matrix([0, 0, 1])
    vc = smp.diff(xc, t)
    vc_carre = vc.T.dot(vc).simplify()
    omega = smp.Matrix(
        [
            phi_d * smp.sin(the) * smp.sin(psi) + the_d * smp.cos(psi),
            phi_d * smp.sin(the) * smp.cos(psi) - the_d * smp.sin(psi),
            phi_d * smp.cos(the) + psi_d,
        ]
    )
    J1, J3 = smp.symbols("J_1, J_3", real=True)
    I = smp.Matrix([[J1, 0, 0], [0, J1, 0], [0, 0, J3]])
    T_rot = smp.Rational(1, 2) * omega.T.dot(I * omega).simplify()
    T_c = smp.Rational(1, 2) * m * vc_carre

    T = T_c + T_rot

    V = m * g * h * smp.cos(the)
    L = T - V
    L = L.simplify()
    LE1 = smp.diff(L, the) - smp.diff(smp.diff(L, the_d), t)
    LE1 = LE1.simplify()

    LE2 = smp.diff(L, phi) - smp.diff(smp.diff(L, phi_d), t)
    LE2 = LE2.simplify()

    LE3 = smp.diff(L, psi) - smp.diff(smp.diff(L, psi_d), t)
    LE3 = LE3.simplify()
    sols = smp.solve(
        [LE1, LE2, LE3], (the_dd, phi_dd, psi_dd), simplify=False, rational=False
    )
    dz1dt_f = smp.lambdify(
        (g, h, m, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[the_dd]
    )
    dthedt_f = smp.lambdify(the_d, the_d)

    dz2dt_f = smp.lambdify(
        (g, h, m, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[phi_dd]
    )
    dphidt_f = smp.lambdify(phi_d, phi_d)

    dz3dt_f = smp.lambdify(
        (g, h, m, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[psi_dd]
    )
    dpsidt_f = smp.lambdify(psi_d, psi_d)

    def dSdt(S, t):
        the, z1, phi, z2, psi, z3 = S
        return [
            dthedt_f(z1),
            dz1dt_f(g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
            dphidt_f(z2),
            dz2dt_f(g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
            dpsidt_f(z3),
            dz3dt_f(g, h, m, J1, J3, the, phi, psi, z1, z2, z3),
        ]

    g, m, h, J1, J3 = params  ######### IMPORTANT

    t = time
    # Initial Condition 1
    ans = odeint(dSdt, y0=CI, t=t)

    path = 0

    the_t = ans.T[0]
    phi_t = ans.T[2]
    psi_t = ans.T[4]
    x_t = np.sin(phi_t) * np.sin(the_t)
    y_t = -np.cos(phi_t) * np.sin(the_t)
    z_t = np.cos(the_t)

    if plot:
        freq_precession, phi0 = Get_frequency(t, phi_t, 1, 0)
        freq_rotation, psi0 = Get_frequency(t, psi_t, CI[-1], 0)

        print("precession frequency =", freq_precession)
        print("rotation frequency =", freq_rotation)
        print(freq_rotation / freq_precession)
        print((J3 * (2 * np.pi * freq_rotation) ** 2) / (m * g * h))

        plt.figure(figsize=[15, 5])
        plt.subplot(1, 2, 1)
        plt.plot(t, the_t * 180 / np.pi)
        plt.ylabel(r"$\theta(t) \; (\mathrm{degrees})$")
        plt.xlabel(r"$t \; (\mathrm{s})$")

        plt.subplot(1, 2, 2)
        plt.plot(t, convert_angle(phi_t))
        plt.ylabel(r"$\phi(t) \; (\mathrm{degrees})$")
        plt.xlabel(r"$t \; (\mathrm{s})$")
        plt.show()

        i = 0
        f = 1000000

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
            x=x_t[i:f],
            y=y_t[i:f],
            z=z_t[i:f],
            mode="lines",
            line=dict(color="green", width=2),
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

    return the_t, phi_t, psi_t, x_t, y_t, z_t, path

def AmplitudeNutation(time, theta):
    """
    *Compute the nutation amplitude.*

    ## Parameters :
    - time (list or np.array) : time list or np.array corresponding to the signal
    - theta (list or np.array) : theta angle list or np.array found according to SolveGyroscope

    ## Return :
    - amplitude (float) : Nutation amplitude
    """

    signalFlat, _ = flatten(time, theta, 1, method='biweight', return_trend=True) # Compute the trend curve and flat o
    dsOriginal = np.max(theta) - np.min(theta)
    dsFlat = np.max(signalFlat) - np.min(signalFlat)
    
    return (np.max(signalFlat) - np.min(signalFlat))*dsOriginal/dsFlat