import gym 
from gym import spaces 
import numpy as np
from gen_data_config import G1, G2, G3, G4, G5
from SynGenLoss_v2.Model1 import CapabilityDiagram 
from SC_create_grid import create_grid
from load_model import PQLoadModel, PQ_model
import pandapower as pp 

def get_load():
    while True: 
        P_vals, Q_vals = PQ_model.sample_PQ()
        for P, Q in zip(P_vals, Q_vals): 
            yield P, Q 

class Env_v1(gym.Env):
    def __init__(self):
        super(Env_v1, self).__init__()

        self.load_model = get_load()
        self.gens = [G1, G2, G3, G4, G5]
        self.CDs = [CapabilityDiagram(G) for G in self.gens]
        self.Sn = np.array([G.md.Sn_mva for G in self.gens])
        self.idx_max = 5
        self.dV_max = 0.03
        self.net = create_grid(1.0, 0.0, 1.0)

        #Reward hyperparameters: (All should be positive)
        self.c1 = 0.0 # Change in input voltage 
        self.c2 = 10.0 # Deviation from Q target 
        self.c3 = 1.0 # Outside allowed voltage range 
        self.c4 = 1.0 # Outside cap diag 
        self.c5 = 1.0 # Efficiency
        self.action_low = -1
        self.action_high = 1
        self.action_space = spaces.Box(low=np.array([self.action_low*2]*5, dtype=np.float32), 
                                       high=np.array([self.action_high*2]*5, dtype=np.float32), dtype=np.float32)
        # self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(5,))

        """Observations: P_gs, V_gs, Q_gs, Q_mins, Q_maxs, V_buses, Q_out, dQ_out -> size = 35"""
        P_g_low = [0.0]*5
        self.V_g_low = [0.9]*5
        Q_g_low = [-1.0]*5
        Q_min_low = [-1.0]*5
        Q_max_low = [-1.0]*5
        V_bus_low = [0.9]*8
        Q_out_low = [-1.0]
        dQ_out_low = [-1.0]
        self.obs_low = np.array(P_g_low + self.V_g_low + Q_g_low + Q_min_low + Q_max_low + V_bus_low + Q_out_low + dQ_out_low)

        P_g_high = [1.0]*5
        self.V_g_high = [1.1]*5
        Q_g_high = [1.0]*5
        Q_min_high = [1.0]*5
        Q_max_high = [1.0]*5
        V_bus_high = [1.1]*8
        Q_out_high = [1.0]
        dQ_out_high = [1.0]
        self.obs_high = np.array(P_g_high + self.V_g_high + Q_g_high + Q_min_high + Q_max_high + V_bus_high + Q_out_high + dQ_out_high)
        # self.observation_space = spaces.Box(low=np.array(obs_low, dtype=np.float32), 
        #                                     high=np.array(obs_high, dtype=np.float32), 
        #                                     dtype=np.float32) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(35,))

    def scale_action(self, action): 
        action = -0.1 + 0.2/(self.action_high-self.action_low) * (action - self.action_low)
        action = np.clip(action, -self.dV_max, self.dV_max)
        return action

    def scale_obs(self, obs): 
        return -1.0 + 2.0/(self.obs_high - self.obs_low) * (obs - self.obs_low)

    def change_net(self, P_out_mva, V_gs, V_out):
        self.P_gs_mva = np.array([G.md.Sn_mva/sum(self.Sn) for G in self.gens])*P_out_mva 
        for i, (P, V) in enumerate(zip(self.P_gs_mva, V_gs)): 
            self.net.gen.at[i, 'p_mw'] = P
            self.net.gen.at[i, 'vm_pu'] = V 
        self.net.ext_grid.at[0, 'vm_pu'] = V_out

    def get_obs(self): 
        pp.runpp(self.net)
        self.P_gs = self.P_gs_mva / self.Sn
        self.Q_gs = np.array([self.net.res_gen.at[i, 'q_mvar'] for i in range(len(self.gens))]) / self.Sn
        self.V_bus = np.array([self.net.res_bus.at[i, 'vm_pu'] for i in range(8)])
        self.P_out = -self.net.res_ext_grid.at[0, 'p_mw']
        self.Q_out = -self.net.res_ext_grid.at[0, 'q_mvar']
        self.dQ_out = (self.Q_out_req - self.Q_out) / sum(self.Sn)
        self.Q_mins, self.Q_maxs = self.get_Q_lims()
        obs = np.concatenate([self.P_gs, self.V_gs, self.Q_gs, self.Q_mins, self.Q_maxs, self.V_bus, [self.Q_out_req / sum(self.Sn), self.dQ_out]], dtype=np.float32)
        return obs

    def get_Q_lims(self):
        Q_mins = np.zeros(len(self.gens), dtype=np.float64) 
        Q_maxs = np.zeros(len(self.gens), dtype=np.float64) 
        for i, (CD, V, P) in enumerate(zip(self.CDs, self.V_gs, self.P_gs)): 
            _, Q_min, Q_max = CD.get_Q_lims(V, P)
            Q_mins[i] = Q_min
            Q_maxs[i] = Q_max
        return Q_mins, Q_maxs
        
    def reset(self):
        self.idx = 0
        # self.V_gs = np.random.uniform(self.V_g_low[0], self.V_g_high[0], 5)
        self.V_gs = np.random.uniform(0.9, 1.05, 5)
        self.P_out_req, self.Q_out_req = next(self.load_model)
        self.change_net(self.P_out_req, self.V_gs, 1.0)  
        obs = self.get_obs()
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.idx += 1
        self.dV = self.scale_action(action)
        self.V_gs += self.dV 
        self.sanity_check() # Checks and corrects unreasonable voltage values. 
        self.change_net(self.P_out_req, self.V_gs, 1.0)
        obs = self.get_obs() 
        obs = np.array(self.scale_obs(obs), dtype=np.float32)
        rew = self.calc_reward()
        done = False if self.idx < self.idx_max else True
        info = {}
        return obs, rew, done, info

    def calc_eff(self): 
        P_loss_grid_mw = self.net.res_line['pl_mw'] + self.net.res_trafo['pl_mw']
        P_loss_gens = 0 
        for G, P, Q, V in zip(self.gens, self.P_gs, self.Q_gs, self.V_gs): 
            n, P_l_s, P_l_r, P_l_c = G.calc_losses_pu(P, Q, V)
            P_loss_gens += (P_l_s + P_l_r + P_l_c)*G.md.Sn_mva
        eff = abs(self.P_out) / (abs(self.P_out) + P_loss_gens)
        return eff 

    def calc_reward(self): 
        R1 = self.c1 * np.sum(np.abs(self.dV)) 
        R2 = self.c2 * abs(self.dQ_out)
        R3 = self.c3 * (len(self.V_gs[np.logical_or(self.V_gs > 1.05, self.V_gs < 0.9)]) + len(self.V_bus[np.logical_or(self.V_bus > 1.05, self.V_bus < 0.95)]))
        R4 = self.c4 * len(self.Q_gs[np.logical_or(self.Q_gs > self.Q_maxs, self.Q_gs < self.Q_mins)])
        R5 = self.c5 * (self.calc_eff() - 0.9)
        return -R1 - R2 - R3 - R4 + R5

    def sanity_check(self): 
        self.V_gs[self.V_gs < 0.7] = 0.7
        self.V_gs[self.V_gs > 1.3] = 1.3 