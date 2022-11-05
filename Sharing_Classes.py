import numpy as np 
from numpy import sin, cos, sqrt
from scipy.interpolate import interp1d 
from scipy.optimize import root 

from typing import Sequence

delta_f = lambda V_hv, V_g, P_g, X_T: np.arcsin(P_g*X_T / (V_hv*V_g))
Q_g_f = lambda V_hv, V_g, P_g, X_T: V_g**2/X_T - V_hv*V_g*cos(delta_f(V_hv, V_g, P_g, X_T))/X_T

def Q_g_HV_f(V_hv, V_g, P_g, X_T): 
    Q_g = Q_g_f(V_hv, V_g, P_g, X_T)
    Q_g_HV = (Q_g*V_g - X_T*(P_g**2 + Q_g**2)) / V_g**2
    return Q_g_HV

class Sharing:
    def __init__(self, x_min, x_max): 
        self.x_min = np.array(x_min) 
        self.x_max = np.array(x_max)
        self.x2 = []
        for i in range(len(x_min)): # Defines all the P2 points for all machines
            x2 = x_min.copy() 
            x2[i] = x_max[i] 
            self.x2.append(x2)
        self._calc_gamma_val()
        self._define_param_curves()
        
    def _calc_gamma_val(self):
        self.gammas = []
        for i in range(len(self.x_min)): 
            dx1 = self.x2[i] - self.x_min 
            dx2 = self.x_max - self.x2[i]
            self.gammas.append( dx1.sum() / (dx1.sum() + dx2.sum()) ) 
            
    def _define_beta_f(self, x_min, x_max, x2_i, gamma_i): 
        def beta_i(gamma): 
            if gamma <= gamma_i: 
                dx1 = x2_i - x_min
                return x_min + dx1*gamma/gamma_i
            elif gamma > gamma_i: 
                dx2 = x_max - x2_i
                return x_max - dx2*(1-gamma)/(1-gamma_i)
        return beta_i
    
    def _define_param_curves(self): 
        self.betas = []
        for i in range(len(self.x_min)): 
            beta_f = self._define_beta_f(self.x_min, self.x_max, self.x2[i], self.gammas[i])
            self.betas.append(beta_f) 
    
class PSharing(Sharing): 
    def __init__(self, P_min, P_max):
        super().__init__(P_min, P_max)
    
    def get_cP_curve(self, P_tot): 
        edges = self.get_edge_points(P_tot) 
        cP_curve = self._define_cP_curve(edges)  
        return cP_curve
    
    def _define_cP_curve(self, edges): 
        def cP_curve(cP): 
            res = edges[0].copy()
            for i, c in enumerate(cP): 
                res += (edges[i+1] - edges[0])*c 
            return res 
        return cP_curve
    
    def get_edge_points(self, P_tot): 
        gamma_est = (P_tot - self.x_min.sum()) / (self.x_max.sum() - self.x_min.sum()) 
        points = [] 
        for i in range(len(self.x_min)): 
            points.append(self.betas[i](gamma_est)) 
        return points

class VSharing(Sharing): 
    def __init__(self, V_min: Sequence[float], V_max: Sequence[float], S_base: Sequence[float]): 
        self.S_base = np.array(S_base)
        super().__init__(V_min, V_max)
        
    def get_edge_points(self, Q_tot, V_hv, P_gs, X_Ts): 
        """Finds edge points for all machines with respect to reactive power sharing. """
        x0 = 0.5
        gammas = []
        for i in range(len(P_gs)):
            sol = root(self._obj_Q_tot, x0, args=(Q_tot, V_hv, P_gs, X_Ts, i))
            gammas.append(sol.x[0])
        edges = [self.betas[i](gamma) for i, gamma in enumerate(gammas)]
        return edges
        
    def _obj_Q_tot(self, gamma, Q_tot, V_hv, P_gs, X_Ts, idx): 
        V_gs = self.betas[idx](gamma) 
        Q_gs = []
        for V_gi, P_gi, X_Ti in zip(V_gs, P_gs, X_Ts): 
            Q_gs.append(Q_g_HV_f(V_hv, V_gi, P_gi, X_Ti)) 
        Q_gs = np.array(Q_gs)*self.S_base
        return Q_tot - np.sum(Q_gs)
        
    def get_cV_curve(self, Q_tot, V_hv, P_gs, X_Ts): 
        """NOTE: Works only for two-machine systems atm."""
        edges = self.get_edge_points(Q_tot, V_hv, P_gs, X_Ts) 
        V_g1_vals = np.linspace(edges[1][0], edges[0][0], 11)
        V_g2_vals = []
        x0 = (self.x_min[1] + self.x_max[1])/2 
        for V_g1 in V_g1_vals:
            sol = root(self._obj_cV_curve, x0, args=(Q_tot, V_hv, P_gs, X_Ts, V_g1))
            V_g2_vals.append(sol.x[0])
        cV_vals = np.linspace(0.0, 1.0, 11)
        V_g1_f = interp1d(cV_vals, V_g1_vals)
        V_g2_f = interp1d(cV_vals, V_g2_vals)
        return lambda cV: np.array([V_g1_f(cV), V_g2_f(cV)])
    
    def _obj_cV_curve(self, V_g2, Q_tot, V_hv, P_gs, X_Ts, V_g1): 
        V_gs = np.array([V_g1, V_g2[0]]) 
        Q_gs = []
        for V_gi, P_gi, X_Ti in zip(V_gs, P_gs, X_Ts): 
            Q_gs.append(Q_g_HV_f(V_hv, V_gi, P_gi, X_Ti)) 
        Q_gs = np.array(Q_gs)*self.S_base
        return Q_tot - np.sum(Q_gs)