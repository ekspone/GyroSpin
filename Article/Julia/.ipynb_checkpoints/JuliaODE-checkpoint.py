import numpy as np
import matplotlib.pyplot as plt
import diffeqpy
from diffeqpy import de, ode


def the_dd_Free(t, Epp, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_):
    the_dd_free = (Epp - p_psi * phi_d + J1_ * de.cos(the) * (phi_d)**2) * de.sin(the) / J1_
    return the_dd_free


def phi_dd_Free(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_):
    phi_dd_free = (-2 * de.cos(the) * phi_d * the_d + p_psi * the_d / J1_) / de.sin(the)
    return phi_dd_free


def psi_dd_Free(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_):
    psi_dd_free = ((1 + de.cos(the)**2) * the_d * phi_d - p_psi * de.cos(the) * the_d / J1_  ) / de.sin(the)
    return psi_dd_free


def the_dd_XY(t, Epp, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, w, p, K):
    the_dd_free = the_dd_Free(t, Epp, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_)
    the_dd_forced = - K * (w**2) * de.sin(w*t+p - phi) * de.cos(the) / J1_
    the_dd = the_dd_free + the_dd_forced
    return the_dd


def phi_dd_XY(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, w, p, K):
    phi_dd_free = phi_dd_Free(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_)
    phi_dd_forced = K * (w**2) * de.cos(w*t+p - phi) / (J1_ * de.sin(the))
    phi_dd = phi_dd_free + phi_dd_forced
    return phi_dd


def psi_dd_XY(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, w, p, K):
    psi_dd_free = psi_dd_Free(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_)
    psi_dd_forced = - K * (w**2) * de.cos(w*t+p - phi) / (J1_ * de.tan(the))
    psi_dd = psi_dd_free + psi_dd_forced
    return psi_dd


def Gyro_Mvt_Free(u, params, t):
    the, the_d, phi, phi_d, psi, psi_d = u
    
    Epp, J1_, p_psi, w, p, K = params
    
    return [
            the_d,
            the_dd_Free(t, Epp, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_),
            phi_d,
            phi_dd_Free(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_),
            psi_d,
            psi_dd_Free(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_),
            ]


def Gyro_Mvt_Forced_XY(u, params, t):
    the, the_d, phi, phi_d, psi, psi_d = u
    
    Epp, J1_, p_psi, w, p, K = params
    
    return [
            the_d,
            the_dd_XY(t, Epp, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, w, p, K),
            phi_d,
            phi_dd_XY(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, w, p, K),
            psi_d,
            psi_dd_XY(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_, w, p, K),
            ]




def Jacobian_Free(u, params, t):

    the, the_d, phi, phi_d, psi, psi_d = u
    
    Epp, J1_, p_psi, w, Phi, K = params
    
    
    L1 = [0, 1, 0, 0, 0, 0]
    L3 = [0, 0, 0, 1, 0, 0]
    L5 = [0, 0, 0, 0, 0, 1]
    psi_norm = p_psi / J1_
    Ep_norm = Ep / J1_
    inverse_sin_carre = 1 / (de.sin(the))**2
    L2 = [Ep_norm * de.cos(the) - psi_norm * de.cos(the) * phi_d + de.cos(2*the) * (phi_d)**2, 0, 0,
         - psi_norm * de.sin(the) + de.sin(2*the) * phi_d, 0, 0]
    L4 = [2 * inverse_sin_carre * the_d * phi_d - psi_norm * de.cos(the) * inverse_sin_carre * the_d,
          -2 * phi_d / de.tan(the) + psi_norm / de.sin(the), 0, -2 * the_d / de.tan(the), 0, 0]
    inverse_tan_carre = 1 / (de.tan(the))**2
    L6 = [-de.cos(the) * (2 + inverse_tan_carre) * the_d * phi_d + psi_norm * the_d * inverse_sin_carre, 
         (1 + de.cos(the)**2) * phi_d / de.sin(the) - psi_norm / de.tan(the), 0,
         (1 + de.cos(the)**2) * the_d / de.sin(the), 0, 0]

    return [L1, L2, L3, L4, L5, L6]


def Jacobian_Forced_XY(u, params, t):

    the, the_d, phi, phi_d, psi, psi_d = u
    
    Epp, J1_, p_psi, w, Phi, K = params
    
    K_w2_norm = K * (w**2) / J1_
    L0 = [0] * 6
    L2_XY = [K_w2_norm * de.sin(the) * de.sin(w*t+p - phi), K_w2_norm * de.cos(the) * de.cos(w*t+p - phi), 0, 0, 0]
    L4_XY = [-K_w2_norm * de.cos(the) * de.cos(w*t+p-phi) / (de.sin(the)**2), 0, 
            K_w2_norm * de.sin(w*t+p - phi) / de.sin(the), 0, 0, 0]
    L6_XY = [K_w2_norm * de.cos(w*t+p - phi) / (de.sin(the)**2), 0, 
            -K_w2_norm * de.sin(w*t+p - phi) / de.tan(the), 0, 0, 0]
    J = Jacobian_Free(t, the, phi, psi, the_d, phi_d, psi_d, p_psi, J1_)
    J[1] = [J[1][i] + L2_XY[i] for i in range(6)] 
    J[3] = [J[3][i] + L4_XY[i] for i in range(6)] 
    J[5] = [J[5][i] + L6_XY[i] for i in range(6)] 
    
    return J





