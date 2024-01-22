import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import odeint
import plotly.graph_objects as go
from IPython.display import HTML
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.animation import FuncAnimation, FFMpegWriter
import scipy.signal as sgn

def convert_angle(angle):
    """
    Convert a angle from radian (rad) to degree (deg).

    ## Pamameters :
    angle : float
        Angle in radian unit

    ## Return : 
    angle : float
        Angle in degree unit
    """

    return np.mod(angle, 2 * np.pi) * 180 / np.pi


def lin(x, a, b):
    """
    Numerical representation of a linea function.

    ## Pamameters :
    x : float
        Real number
    a : float
        Function's slope
    b : float
        Function's intercept

    ## Return : 
    y : float
        return y = a*x + b
    """

    return a * x + b


def Get_frequency(t, angle, freq0, angle0):
    """
    Compute the oscilation frequency.

    ## Pamameters
    `t` : list or np.array
        Time array (Same size than `angle`)
    `angle` : float
        Angle array computed with Solve_Gyro (Same size than `t`)
    `freq0` : float
        Initial guess for the frequence to perform the curve fitting
    `angle0` : float
        Initial guess for the angle-intercept to perform the curve fitting

    ## Return
    `freq` : float
        Angle oscilation frequency
    `phi0` :  float
        Angle intercept at t = 0s
    """

    tab, pcov = curve_fit(lin, t, angle, p0=[freq0, angle0])
    freq = tab[0] / (2 * np.pi)
    phi0 = tab[1]
    return freq, phi0


def MatrixBasisChange(theta, phi, psi):
    """
    Compute the matrix of base change from the cartesian coordinate to the Euler angle coordinate

    ## Pamameters
    `theta` : list or np.array
        Theta angle value across time 
    `phi` : list or np.array
        Phi angle value across time 
    `psi` : list or np.array
        Psi angle value across time 


    ## Return
    `R` : MutableDenseMatrix
        Change of base matrix
    """

    R3 = smp.Matrix([[smp.cos(psi),-smp.sin(psi),0],
                    [smp.sin(psi),smp.cos(psi),0],
                    [0,0,1]])

    R2 = smp.Matrix([[1,0,0],
                    [0,smp.cos(theta),-smp.sin(theta)],
                    [0,smp.sin(theta),smp.cos(theta)]])

    R1 = smp.Matrix([[smp.cos(phi),-smp.sin(phi),0],
                    [smp.sin(phi),smp.cos(phi),0],
                    [0,0,1]])

    R = R1*R2*R3

    return R


def ParametersDefinition():
    """
    Define all the parameters in the Sympy format

    ## Return
    All the parameters and function in the Sympy format

    """

    t, h, g, m, x0, p, w, J1, J3 = smp.symbols(r't h g m x_0 \Phi \omega, J_1, J_3', real=True)
    the, phi, psi = smp.symbols(r'\theta \phi \psi', cls=smp.Function)

    # Function
    the = the(t)
    phi = phi(t)
    psi = psi(t)

    # First order derivatives
    the_d = smp.diff(the,t)
    phi_d = smp.diff(phi,t)
    psi_d = smp.diff(psi,t)

    # Second order derivatives
    the_dd = smp.diff(the_d,t)
    phi_dd = smp.diff(phi_d,t)
    psi_dd = smp.diff(psi_d,t)

    return t, h, g, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d, the_dd, phi_dd, psi_dd


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

def MergeDicts(list_of_dicts):
    result_dict = {}
    for d in list_of_dicts:
        result_dict.update(d)
    return result_dict

def GenMomentum(lagrangian, genCoordinate, genSpeed, mode = "equation"):
    momentumName = smp.symbols('p_{}'.format(genCoordinate.name))
    if mode == "name":
        return momentumName
    
    momentum = smp.diff(lagrangian, genSpeed)
    return momentumName - momentum

def Hamiltonian(lagrangian, genCoordinates, genSpeeds):
    n = len(genCoordinates)
    rules = MergeDicts(sum([smp.solve(GenMomentum(lagrangian, genCoordinates[i], genSpeeds[i]), genSpeeds[i], dict=True) for i in range(n)], []))
    return (sum([GenMomentum(lagrangian, genCoordinates[i], genSpeeds[i], mode="name") * genSpeeds[i] for i in range(n)]) - lagrangian).xreplace(rules)  

def PlottingGyroscope(time, theta, phi, x, y, z):
    """
    Plot the gyroscope main data (theta angle, phi angle, position the carthesian coordinate system)

    ## Parameters :
    `time` : list or np.array
        Time array (Same size than `angle`)
    `theta` : list or np.array
        Theta angle value across time
    `phi` : list or np.array
        Phi angle value across time
    `x` : list or np.array
        X-axis position of the gyroscope
    `y` : list or np.array
        Y-axis position of the gyroscope
    `z` : list or np.array
      Z-axis position of the gyroscope
    """

    plt.figure(figsize=[15, 5], dpi=300)

    plt.subplot(1, 2, 1)
    plt.plot(time, convert_angle(theta))
    plt.ylabel(r"$\theta(t) \; (\mathrm{deg})$")
    plt.xlabel(r"$t \; (\mathrm{s})$")

    plt.subplot(1, 2, 2)
    plt.plot(time, convert_angle(phi))
    plt.ylabel(r"$\phi(t) \; (\mathrm{deg})$")
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


def NumbersGyroscope(time, phi, psi, params, CI):
    """
    ## Parameters :
    `time` : list or np.array
        Time array (Same size than `angle`)
    `theta` : list or np.array
        Theta angle value across time
    `phi` : list or np.array
        Phi angle value across time

    """
    g, m, h, J1, J3, x0, p, w = params

    freq_precession, _ = Get_frequency(time, phi, 1, 0)
    freq_rotation, _ = Get_frequency(time, psi, CI[-1], 0)
    GyrFreq = freq_rotation / freq_precession
    GyrParams = J3 * (2 * np.pi * freq_rotation) ** 2 / (m * g * h)
    GyrGap = np.abs((GyrFreq - GyrParams) / (GyrParams)) * 100
    Larmor_freq = (m * g * h) / (J3 * freq_rotation * (2 * np.pi)**2)

    print("Precession frequency =", freq_precession)
    print("Rotation frequency =", freq_rotation)
    print("Gyros number (frequency) =", GyrFreq)
    print("Gyros number (parameters) =", GyrParams)
    print("Gyros number gap = {} %".format(GyrGap))
    print("Larmor frequency = ", Larmor_freq, ' Hz')


def MatrixForcing(x0, p, w, t, forcing="free"):
    """
    Return the matrix corresponding to the forcing

    `x0` : sympy.Symbolsympy.Symbol
        Excitation amplitude symbol
    `w` : sympy.Symbol
        Excitation Pulsation symbol
    `t` : sympy.Symbol
        Time symbol
    `forcing` : bool (default : `"free"`)
        Excitation forcing mode
            - `"free"` : free motion (motor turned off)
            - `"X"` : excitation along X-axis
            - `"Y"` : excitation along Y-axis
            - `"XY"` : excitation along X-axis and Y-axis with a 90 deg phase change

    return sympy.Symbol
    """

    if forcing == "free":
        return smp.Matrix([0, 0, 0])

    elif forcing == "X":
        return smp.Matrix([x0 * smp.cos(w * t + p), 0, 0])

    elif forcing == "Y":
        return smp.Matrix([0, x0 * smp.cos(w * t + p), 0])

    if forcing == "XY":
        return smp.Matrix([x0 * smp.cos(w * t + p), x0 * smp.sin(w * t + p), 0])


def Solve_Gyro(time, CI, params, forcing="free", plot=True, numbers=True):
    """
    Compute the numerical solution of the gyroscope motion for the given parameters and boundary conditions

    ## Parameters
    `time` : list or np.array
        Time array
    `CI` : list or np.array
        Boundary conditions for the Euler angle. In the format [theta(t = 0), phi(t = 0), psi(t = 0), dtheta(t = 0)/dt, dphi(t = 0)/dt, dpsi(t = 0)/dt,]
    `params` : list or np.array
    `forcing` : str
        Forcing mode (see the the function `MatrixForcing`)
    plot : bool (default : `True`)
        If `True` return the gyroscope plot
    numbers : bool (default : `True`)
         If `True` return the gyroscope caracteristic numbers

    ## Return
    `the_t`: np.array
        Solution for theta angle solution across time
    `phi_t`: np.array
        Solution for Phi angle solution across time
    `psi_t` : np.array
        Solution for Phi angle solution across time
    `the_t_d` : np.array
        Derivative of the solution for theta angle solution across time
    `phi_t_d` : np.array
        Derivative of the solution for phi angle solution across time
    `psi_t_d` : np.array
        Derivative of the solution for phi angle solution across time
    `x_t` : list or np.array
        Solution for X-axis position of the gyroscope
    `y_t` : list or np.array
        Solution for Y-axis position of the gyroscope
    `z_t` : list or np.array
        Solution for Z-axis position of the gyroscope
    `path` :

    """

    # Parameters settings
    (
        t,
        h,
        g,
        m,
        x0,
        p,
        w,
        J1,
        J3,
        the,
        phi,
        psi,
        the_d,
        phi_d,
        psi_d,
        the_dd,
        phi_dd,
        psi_dd,
    ) = ParametersDefinition()

    parametersSymbol = [t, h, g, m, x0, p, w, J1, J3]
    genCoordinates = [the, phi, psi]
    genSpeeds = [the_d, phi_d, psi_d]
    genAcceleration = [the_dd, phi_dd, psi_dd]

    R = MatrixBasisChange(the, phi, psi)

    # Center of mass position, speed, and speed norm squared
    xc = h * R @ smp.Matrix([0, 0, 1]) + MatrixForcing(x0, p, w, t, forcing)
    xc.simplify()
    vc = smp.diff(xc, t)
    vc.simplify()
    vc_carre = vc.T.dot(vc)
    vc_carre.simplify()

    # Rotation vector
    omega = smp.Matrix(
        [
            phi_d * smp.sin(the) * smp.sin(psi) + the_d * smp.cos(psi),
            phi_d * smp.sin(the) * smp.cos(psi) - the_d * smp.sin(psi),
            phi_d * smp.cos(the) + psi_d,
        ]
    )

    # Inertia matrix
    I = smp.Matrix([[J1, 0, 0], [0, J1, 0], [0, 0, J3]])

    # Kinetic, and potential energy
    T_rot = (
        smp.Rational(1, 2) * omega.T.dot(I * omega).simplify()
    )  # Rotational kinetic energy
    T_c = smp.Rational(1, 2) * m * vc_carre  # Translational kinetic energy
    T = T_c + T_rot  # Total kinetic energy
    V = m * g * h * smp.cos(the)  # Potential energy (gravitation)

    # Lagrangian
    L = T - V
    L = L.simplify()

    # Euler-Lagrange equation et solving
    equations = [EulerLagrange(L, genCoordinates[i], genSpeeds[i], t) for i in range(3)]
    sols = smp.solve(
        equations, (the_dd, phi_dd, psi_dd), simplify=False, rational=False
    )

    dz1dt_f = smp.lambdify(
        (t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[the_dd]
    )
    dthedt_f = smp.lambdify(the_d, the_d)

    dz2dt_f = smp.lambdify(
        (t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[phi_dd]
    )
    dphidt_f = smp.lambdify(phi_d, phi_d)

    dz3dt_f = smp.lambdify(
        (t, g, h, m, x0, p, w, J1, J3, the, phi, psi, the_d, phi_d, psi_d), sols[psi_dd]
    )
    dpsidt_f = smp.lambdify(psi_d, psi_d)

    def dSdt(S, t):
        the, z1, phi, z2, psi, z3 = S
        return [
            dthedt_f(z1),
            dz1dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
            dphidt_f(z2),
            dz2dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
            dpsidt_f(z3),
            dz3dt_f(t, g, h, m, x0, p, w, J1, J3, the, phi, psi, z1, z2, z3),
        ]

    g, m, h, J1, J3, x0, p, w = params
    w = 2 * np.pi * w
    t = time
    ans = odeint(dSdt, y0=CI, t=t)

    path = 0

    the_t = ans.T[0]
    phi_t = ans.T[2]
    psi_t = ans.T[4]

    the_t_d = ans.T[1]
    phi_t_d = ans.T[3]
    psi_t_d = ans.T[5]

    x_t = np.sin(phi_t) * np.sin(the_t)
    y_t = -np.cos(phi_t) * np.sin(the_t)
    z_t = np.cos(the_t)

    if numbers:
        NumbersGyroscope(t, phi_t, psi_t, params, CI)

    if plot:
        path = PlottingGyroscope(t, the_t, phi_t, x_t, y_t, z_t)

    return the_t, phi_t, phi_t, the_t_d, phi_t_d, psi_t_d, x_t, y_t, z_t, path


def Solve_Gyro_Free(time, CI, params, plot=True, numbers=True):
    """Wrapping of `Solve_Gyro` for the free motion"""
    return Solve_Gyro(time, CI, params, forcing="free", plot=plot, numbers=numbers)


def Solve_Gyro_Forced_XY(time, CI, params, plot=True, numbers=True):
    """Wrapping of `Solve_Gyro` for the motion with a excitation along X and Y-axis (circular translation)"""
    return Solve_Gyro(time, CI, params, forcing="XY", plot=plot, numbers=numbers)


def Solve_Gyro_Forced_X(time, CI, params, plot=True, numbers=True):
    """Wrapping of `Solve_Gyro` for the motion with a excitation along X-axis"""
    return Solve_Gyro(time, CI, params, forcing="X", plot=plot, numbers=numbers)


def Solve_Gyro_Forced_Y(time, CI, params, plot=True, numbers=True):
    """Wrapping of `Solve_Gyro` for the motion with a excitation along Y-axis"""
    return Solve_Gyro(time, CI, params, forcing="Y", plot=plot, numbers=numbers)


###### IN PROGRESS (MABYE A PROBLEM IN THE JUnCTION) #####
'''
def SwitchForcing(solution, newForcing, newTimeArray, params):
    the_t, phi_t, phi_t, the_t_d, phi_t_d, psi_t_d, _, _, _, _ = solution
    newCI = [the_t[-1], phi_t[-1], phi_t[-1], the_t_d[-1], phi_t_d[-1], psi_t_d[-1]]
    newSolution = Solve_Gyro(
        newTimeArray, newCI, params, forcing=newForcing, plot=False, numbers=False
    )
    return (np.concatenate(solution[i], newSolution[i]) for i in range(10))


def ChangeForcing(CI, params, listForcing, timeSettings):
    time = np.linspace(*timeSettings[0])
    solution = Solve_Gyro(time, CI, params, forcing="free", plot=False, numbers=False)

    for i in range(1, len(listForcing)):
        solution = SwitchForcing(solution, listForcing[i], timeSettings[i], params)

    return solution
'''

###### IN PROGRESS (MABYE A PROBLEM IN THE JUnCTION) #####


def Stop_Forcing(t1, t2, theta, phi, psi, theta_D, phi_D, psi_D, params, bool_plot=False):
    
    CI = [theta[-1], theta_D[-1], phi[-1], phi_D[-1], psi[-1], psi_D[-1]]

    t = np.linspace(0, t2, 1000, endpoint=True)

    return Solve_Gyro_Free(t, CI, params, plot=bool_plot)


def Get_Gyro_Position(t1, t2, CI, params, bool_plot=False):
    """
    Compute the position of the in the cartesian coordinate system
    """
    T1 = np.linspace(0, t1, 1000, endpoint=True)
    T2 = np.linspace(0, t2, 1000, endpoint=True) + t1

    the_f, phi_f, psi_f, theD_f, phiD_f, psiD_f, _, _, _, path_f = Solve_Gyro_Forced_XY(
        T1, CI, params, plot=bool_plot
    )

    the_t_lib, phi_t_lib, psi_t_lib, x_t_lib, y_t_lib, z_t_lib, _, _, _, path_lib = Stop_Forcing(
        t1, t2, the_f, phi_f, psi_f, theD_f, phiD_f, psiD_f, params, bool_plot
    )

    T = np.concatenate((T1, T2))
    THETA = np.concatenate((the_f, the_t_lib))
    PSI = np.concatenate((psi_f, psi_t_lib))
    PHI = np.concatenate((phi_f, phi_t_lib))

    X = np.sin(PHI) * np.sin(THETA)
    Y = -np.cos(PHI) * np.sin(THETA)
    Z = np.cos(THETA)

    return T, X, Y, Z, THETA, PHI, PSI


def PlotNutation(time, theta, thetaFlat, nutation):
    """
    Plot the nutation angle

    ## Parameters :
    `time` : list, np.araay
        Time array
    `theta` : list, np.array
        Theta array across time
    `thetaFlat` :
        Flatted `theta` angle
    `nutation` :
        Nutation angle
    """

    plt.figure(figsize=(9, 3), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(time, theta)
    plt.plot(time, thetaFlat, linestyle="--")
    plt.subplot(1, 2, 2)
    plt.plot(time, nutation)
    plt.show()


def Nutation(time, theta, free, window_length=20, polyorder=2, plot=False, data=False):
    """
    Compute the nutation and its amplitude. This function is a wrapping of 'scipy.signal.savgol_filter'

    ## Parameters :
    time : list or numpy.array (1D)
        Range of time. Same size than `theta`
    theta : list or numpy.array (1D)
        Theta angle. Same size than `time`
    free : bool :
        `True` if the gyroscope has a free motion, False otherwise
    window_length : int, optional
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int, optional
        The order of the polynomial used to fit the samples. polyorder must be less than `window_length`.
    plot : bool, optional
        If `True`, nutation graph is plotted
    data : bool, , optional
        If `True`, the function return the nutation angle and the time range

    ## Return :
    amplitudeNutation : float :
        Nutation amplitude (maximum - minimum)
    time : list or numpy.array (1D)
        Range of time.
    nutation : numpy.array (1D)
        Nutation angle
    """

    if free:
        if plot:
            PlotNutation(time, theta, np.mean(theta) + np.zeros(time.shape), theta)
        if data:
            return time, theta

        return np.max(theta) - np.min(theta)

    thetaFlat = sgn.savgol_filter(theta, window_length, polyorder)
    nutation = theta - thetaFlat
    amplitudeNutation = np.max(nutation) - np.min(nutation)

    if plot:
        PlotNutation(time, theta, thetaFlat, nutation)

    if data:
        return time, nutation

    return amplitudeNutation
