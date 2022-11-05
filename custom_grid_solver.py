from scipy.optimize import root 
import numpy as np
import cmath as cm 
import SynGenLoss_v2 as sgl 
from SynGenLoss_v2.Model1 import LineDataClass, LineModel
from gen_data_config import G1, G2, G3, G4, G5, T1, T2, T3, T4, T5, L1, L2, L3, L4, L5, L6, L7
from copy import deepcopy
from numba import njit 
from environment2_v2 import Env_v2, TestEnv 
import pandapower as pp


def create_y_bus(): 
    lines = [L1, L2, L3, L4, L5, L6, L7]
    lines_loc = [(0, 1), (1, 4), (1, 2), (2, 5), (2, 3), (3, 6), (3, 7)]
    trafos = [T1, T2, T3, T4, T5] 
    trafos_loc = [(1, 8), (4, 9), (5, 10), (6, 11), (7, 12)]
    Y_bus = np.zeros((13, 13), dtype=np.complex64) 
    shunt_array = np.zeros(13, dtype=np.complex64)
    for L, L_loc in zip(lines, lines_loc): 
        i1 = L_loc[0] 
        i2 = L_loc[1] 
        Y_bus[i1, i2] = Y_bus[i2, i1] = -(L.R + 1j*L.X)**-1 
        shunt_array[i1] += L.Y/2  
        shunt_array[i2] += L.Y/2
        
    for T, T_loc in zip(trafos, trafos_loc): 
        i1 = T_loc[0] 
        i2 = T_loc[1] 
        Z_base_hv = T1.md.V_hv_kv**2/T1.md.Sn_mva 
        Z_base_lv = T1.md.V_lv_kv**2/T1.md.Sn_mva 
        Y_bus[i1, i2] = Y_bus[i2, i1] = -(T.md.R_T*Z_base_hv + 1j*T.md.X_T*Z_base_hv)**-1 
        shunt_array[i1] += (T.md.Y_E/2) / Z_base_hv
        shunt_array[i2] += (T.md.Y_E/2) / Z_base_hv
        
    for i in range(len(Y_bus)): 
        Y_bus[i, i] = -sum(Y_bus[i, :]) - shunt_array[i]
    return Y_bus, shunt_array 


class PowerSystem: 
    S_g_vals = np.array([G1.md.Sn_mva, G2.md.Sn_mva, G3.md.Sn_mva, G4.md.Sn_mva, G5.md.Sn_mva])
    S_base = sum(S_g_vals)
    V_base = 132 
    Z_base = V_base**2/S_base 
    Y_base = 1/Z_base
    I_base = S_base*1000/V_base / np.sqrt(3)

    Y_bus, shunts = create_y_bus()
    Y_bus_pu = Y_bus / Y_base 
    G_bus_pu = Y_bus_pu.real 
    B_bus_pu = Y_bus_pu.imag 
    shunts_pu = shunts / Y_base 
    
    def do_pf(self, V_gs, P_gs_mva, V_ext): 
        P_gs = P_gs_mva / self.S_base
        delta_0 = np.zeros(12) 
        V_0 = np.ones(7)
        X0 = np.concatenate([delta_0, V_0])
        sol = root(objective, X0, args=(V_gs, P_gs, V_ext, self.G_bus_pu, self.B_bus_pu), tol=1e-6) 
        deltas, V_buses = self.collect_sol(sol)
        delta_vals, V_vals = collect_V_deltas(V_gs, V_buses, V_ext, deltas)
        S_vals = do_pf(V_vals, delta_vals, self.Y_bus_pu) 
        S_vals_mva = S_vals * self.S_base 
        Q_gs_mva = S_vals_mva.imag[-5:]
        P_out_mva = -S_vals_mva.real[0]
        Q_out_mva = -S_vals_mva.imag[0]
        return P_out_mva, Q_out_mva, Q_gs_mva, V_buses
    
    def collect_sol(self, sol):
        res_vals = sol.x 
        deltas = res_vals[:12]
        V_buses = res_vals[12:] 
        return deltas, V_buses

@njit
def do_pf(V_vals, delta_vals, Y_bus_pu): 
    V_vals_c = np.ones(len(V_vals), dtype=np.complex64)
    for i in range(len(V_vals)): 
        V_vals_c[i] = V_vals[i]*np.cos(delta_vals[i]) + 1j*V_vals[i]*np.sin(delta_vals[i])
    I_vals_c = Y_bus_pu @ V_vals_c  
    S_vals = V_vals_c * I_vals_c.conj() 
    return S_vals

@njit
def collect_V_deltas(V_gs, V_buses, V_ext, deltas): 
    delta_vals = np.zeros(13, dtype=np.float64) 
    delta_vals[1:] = deltas
    V_vals = np.ones(13, dtype=np.float64) 
    V_vals[0] = V_ext
    V_vals[1:-5] = V_buses 
    V_vals[-5:] = V_gs 
    return delta_vals, V_vals
    
@njit
def objective(X, V_gs, P_gs, V_ext, G_bus_pu, B_bus_pu): 
    deltas = X[0:12]
    V_buses = X[12:20]
    
    delta_vals, V_vals = collect_V_deltas(V_gs, V_buses, V_ext, deltas)

    P_knowns = np.zeros(13, dtype=np.float64) 
    P_knowns[-5:] += P_gs
    Q_knowns = np.zeros(13, dtype=np.float64)

    P_flow = np.zeros(13, dtype=np.float64)
    Q_flow = np.zeros(13, dtype=np.float64)
    for i in range(0, 13, 1):
        P = V_vals[i]**2 * G_bus_pu[i, i] 
        Q = -V_vals[i]**2 * B_bus_pu[i, i] 
        for k in range(0, 13, 1): 
            if k != i: 
                delta_ik = delta_vals[i] - delta_vals[k]
                P = P + V_vals[i]*V_vals[k]*(B_bus_pu[i, k]*np.sin(delta_ik) + G_bus_pu[i, k]*np.cos(delta_ik))
                Q = Q + V_vals[i]*V_vals[k]*(G_bus_pu[i, k]*np.sin(delta_ik) - B_bus_pu[i, k]*np.cos(delta_ik))
        P_flow[i] = P 
        Q_flow[i] = Q 
        
    P_diff = P_flow[1:] - P_knowns[1:]
    Q_diff = Q_flow[1:8] - Q_knowns[1:8]
    
    return_vals = np.zeros(len(P_diff) + len(Q_diff), dtype=np.float64) 
    return_vals[:len(P_diff)] = P_diff 
    return_vals[len(P_diff):] = Q_diff
    return return_vals 