import pandas as pd 
import numpy as np 
from scipy.interpolate import interp1d
import pandapower as pp 
from SC_create_grid import create_grid

data = pd.read_excel("load_data.xlsx")
data['Hours'] = data['Hours'].map(lambda s: int(s[:2]))
data["NO1_diff"] = np.append([0], np.diff(data["NO1"]) ) 

hour_mean_NO1 = np.array([data[data['Hours']==i]["NO1"].mean() for i in range(24)])
hour_std_NO1 = np.array([data[data['Hours']==i]["NO1"].std() for i in range(24)])
hour_dmean_NO1 = np.array([data[data['Hours']==i]["NO1_diff"].mean() for i in range(24)])
hour_dstd_NO1 = np.array([data[data['Hours']==i]["NO1_diff"].std() for i in range(24)])

def stoc_model(mean_vals, std_vals, diff_mean, diff_std, dt, theta, P_max): 
    def func(N_days=1): 
        x0 = np.random.uniform(mean_vals[0]-1.5*std_vals[0], mean_vals[0]+1.5*std_vals[0])
        x = [x0]
        for i in range(24*N_days):
            dxdt = np.random.normal(diff_mean[i%24], diff_std[i%24]) + theta*(mean_vals[i%24] - x[-1])
            x.append(x[-1] + dxdt * dt) 
        return np.array(x[1:])
    return func 

load_model = stoc_model(hour_mean_NO1, hour_std_NO1, hour_dmean_NO1, hour_dstd_NO1, 1.0, 0.02, P_max=300)

class PQLoadModel: 
    def __init__(self, P_max, P_vals, Q_min_vals, Q_max_vals): 
        """P_max: How the active power is scaled from the previous max of 8000 MW \n 
        P_vals: The values of P_out corresponding to Q_min_vals and Q_max_vals. \n 
        Q_min/max_vals: Reactive power capability with respect to P_vals. """
        self.P_max = P_max 
        self.P_vals = P_vals 
        self.Q_min_vals = Q_min_vals
        self.Q_max_vals = Q_max_vals 
        
        self.Q_f = self.create_Q_norm() 
        self.load_model = stoc_model(hour_mean_NO1, hour_std_NO1, hour_dmean_NO1, hour_dstd_NO1, 1.0, 0.02, P_max=300)
        
    def create_Q_norm(self): 
        self.Q_min_f = interp1d(self.P_vals, self.Q_min_vals)
        self.Q_max_f = interp1d(self.P_vals, self.Q_max_vals)
        self.Q_mid = lambda P: (self.Q_max_f(P) + self.Q_min_f(P))/2
        def Q_sample(P): 
            Q_val = np.random.normal(self.Q_mid(P), self.Q_max_f(P) - self.Q_mid(P))
            return Q_val 
        return Q_sample 
    
    def stoc_model(mean_vals, std_vals, diff_mean, diff_std, dt, theta, P_max): 
        def func(N_days=1): 
            x0 = np.random.uniform(mean_vals[0]-1.5*std_vals[0], mean_vals[0]+1.5*std_vals[0])
            x = [x0]
            for i in range(24*N_days):
                dxdt = np.random.normal(diff_mean[i%24], diff_std[i%24]) + theta*(mean_vals[i%24] - x[-1])
                x.append(x[-1] + dxdt * dt) 
            return np.array(x[1:])
        return func 
    
    def sample_PQ(self): 
        P_vals = self.load_model() * self.P_max/8000
        Q_vals = np.zeros_like(P_vals)
        for i, P in enumerate(P_vals): 
            Q_vals[i] = self.Q_f(P) 
        return P_vals, Q_vals 
        
        
P_pu_vals = np.linspace(0, 1, 10)
V_g_min = 0.9
V_g_max = 1.05

P_min_vals = [] 
P_max_vals = []
Q_min_vals = []
Q_max_vals = []

for P_pu in P_pu_vals: 
    net = create_grid(1.0, V_g=V_g_min, P_pu=P_pu)
    pp.runpp(net) 
    Q_min_vals.append(net.res_bus.iloc[0]["q_mvar"]) 
    P_min_vals.append(net.res_bus.iloc[0]["p_mw"]) 
    net = create_grid(1.0, V_g=V_g_max, P_pu=P_pu)
    pp.runpp(net) 
    Q_max_vals.append(net.res_bus.iloc[0]["q_mvar"]) 
    P_max_vals.append(net.res_bus.iloc[0]["p_mw"]) 
    
V_vals = np.linspace(V_g_min, V_g_max, 10)
P_v_vals = [] 
Q_v_vals = []
for V_g in V_vals: 
    net = create_grid(1.0, V_g=V_g, P_pu=P_pu_vals[-1])
    pp.runpp(net) 
    Q_v_vals.append(net.res_bus.iloc[0]["q_mvar"]) 
    P_v_vals.append(net.res_bus.iloc[0]["p_mw"])

PQ_model = PQLoadModel(max(P_min_vals), P_min_vals, Q_min_vals, Q_max_vals)