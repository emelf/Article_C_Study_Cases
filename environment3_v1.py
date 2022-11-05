import gym 
from gym import spaces 
import numpy as np
from gen_data_config import G1, G2, G3, G4, G5
from SynGenLoss_v2.Model1 import CapabilityDiagram, SimpleCapDiag
from custom_grid_solver import PowerSystem 
from Sharing_Classes import PSharing 
from scipy.optimize import root, root_scalar
import copy 

class Env_v3(gym.Env):
    def __init__(self):
        super(Env_v3, self).__init__()

        self.load_model = InitGrid()
        self.gens = [G1, G2, G3, G4, G5]
        self.CDs = [CapabilityDiagram(G) for G in self.gens]
        self.CDs = [SimpleCapDiag(CD) for CD in self.CDs]
        self.Sn = np.array([G.md.Sn_mva for G in self.gens])
        self.net = PowerSystem()
        self.share = PSharing(np.zeros(len(self.Sn)), self.Sn)
        
        self.N_steps_max = 50 - 1

        #Reward hyperparameters: (All should be positive)
        self.failure = -100
        self.step_reward = -0.0
        self.eff_reward = lambda eff: (eff - 0.90)*100
        self.V_out_binary = -1.0
        self.V_out_weight = -10
            
        self.dQ_threshold = 1.0
        
        self.action_low = -1
        self.action_high = 1
        self.action_space = spaces.Box(low=np.array([-np.inf]*5, dtype=np.float32), 
                                       high=np.array([np.inf]*5, dtype=np.float32), dtype=np.float32)
        self.dV_min = -0.05
        self.dV_max = 0.05

        P_g_low = [0.0]*5
        self.V_g_low = [0.9]*5
        Q_g_low = [-G.md.Sn_mva for G in self.gens]
        Q_out_low = [-sum([G.md.Sn_mva for G in self.gens])]
        Q_req_low = Q_out_low
        dQ_out_low = [-100]
        V_ext_low = [0.9]
        # self.obs_env_low = np.array(P_g_low + Q_g_low + self.V_g_low + Q_out_low + Q_req_low + dQ_out_low + V_ext_low, dtype=np.float32)
        self.obs_env_low = np.array(Q_out_low + Q_req_low + dQ_out_low + V_ext_low, dtype=np.float32)
        self.obs_agent_low = -np.ones(len(self.obs_env_low), dtype=np.float32)
        
        P_g_high = [G.md.Sn_mva for G in self.gens]
        self.V_g_high = [1.1]*5
        Q_g_high = [G.md.Sn_mva for G in self.gens]
        Q_out_high = [sum([G.md.Sn_mva for G in self.gens])]
        Q_req_high = Q_out_high
        dQ_out_high = [100]
        V_ext_high = [1.1]
        
        # self.obs_env_high = np.array(P_g_high + Q_g_high + self.V_g_high + Q_out_high + Q_req_high + dQ_out_high + V_ext_high, dtype=np.float32)
        self.obs_env_high = np.array(Q_out_high + Q_req_high + dQ_out_high + V_ext_high, dtype=np.float32)
        self.obs_agent_high = np.ones(len(self.obs_env_high), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_agent_low, high=self.obs_agent_high, shape=self.obs_env_low.shape)
        
    def V_out_reward(self):
        Vg_low = np.array(self.V_g_low)
        Vg_high = np.array(self.V_g_high) 
        too_low = self.V_gs < Vg_low 
        too_high = self.V_gs > Vg_high 
        reward_low = self.V_out_weight*sum(too_low*(Vg_low - self.V_gs)) + self.V_out_binary*sum(too_low)
        reward_high = self.V_out_weight*sum(too_high*(self.V_gs - Vg_high)) + self.V_out_binary*sum(too_high)
        return reward_low + reward_high

    def scale_action(self, action_agent): 
        action_env = (self.dV_max-self.dV_min)/(self.action_high - self.action_low) * (action_agent - self.action_low) + self.dV_min
        return action_env

    def scale_obs(self, obs_env): 
        return (self.obs_agent_high - self.obs_agent_low)/(self.obs_env_high - self.obs_env_low) * (obs_env - self.obs_env_low) + self.obs_agent_low
    
    def rescale_obs(self, obs_agent):
        return (obs_agent - self.obs_agent_low)*(self.obs_env_high - self.obs_env_low)/(self.obs_agent_high - self.obs_agent_low) + self.obs_env_low
        
    def set_net(self, P_gs_mva=None, V_gs=None, V_ext=None, Q_out_req=None):
        if not P_gs_mva is None: 
            self.P_gs_mva = P_gs_mva
        if not V_gs is None: 
            self.V_gs = V_gs 
        if not V_ext is None: 
            self.V_ext = V_ext 
        if not Q_out_req is None: 
            self.Q_out_req = Q_out_req

    def get_obs(self): 
        self.P_out, self.Q_out, self.Q_gs_mva, self.V_bus = self.net.do_pf(self.V_gs, self.P_gs_mva, self.V_ext)
        self.dQ_out = (self.Q_out_req - self.Q_out)
        # obs = np.concatenate([self.P_gs_mva, self.Q_gs_mva, self.V_gs, [self.Q_out, self.Q_out_req, self.dQ_out, self.V_ext]], dtype=np.float32)
        obs = np.concatenate([[self.Q_out, self.Q_out_req, self.dQ_out, self.V_ext]], dtype=np.float32)
        return obs

    def get_Q_lims(self):
        Q_mins = np.zeros(len(self.gens), dtype=np.float64) 
        Q_maxs = np.zeros(len(self.gens), dtype=np.float64) 
        for i, (CD, V, P, Sn) in enumerate(zip(self.CDs, self.V_gs, self.P_gs_mva, self.Sn)): 
            # _, Q_min, Q_max = CD.get_Q_lims(V, P)
            Q_min, Q_max = CD.get_Q_lims(P/Sn)
            Q_mins[i] = Q_min * Sn
            Q_maxs[i] = Q_max * Sn
        return Q_mins, Q_maxs
    
    def find_Vgs(self, V_gs): 
        V_gs_vec = np.ones(5)*V_gs
        P_out, Q_out, Q_gs, V_buses = self.net.do_pf(V_gs_vec, self.P_gs_mva, self.V_ext)
        return (Q_out - self.Q_out_req)
        
    def reset(self):
        self.episode_step = 0
        self.P_out_req, self.Q_out_req, self.V_ext, self.P_gs_mva = self.load_model.get_grid_data(self)
        self.V_gs = np.random.uniform(0.9, 1.05, size=5)
        self.set_net(self.P_gs_mva, self.V_gs, self.V_ext, self.Q_out_req)  
        obs_env = self.get_obs()
        obs_agent = np.array(self.scale_obs(obs_env), dtype=np.float32)
        return obs_agent

    def step(self, action_agent):
        action_env = self.scale_action(action_agent)
        self.dV = np.clip(action_env, self.dV_min, self.dV_max)
        self.V_gs = self.V_gs + self.dV
        self.V_gs = np.clip(self.V_gs, 0.6, 1.4)

        self.set_net(V_gs=self.V_gs) 
        obs_env = self.get_obs() 
        obs_agent = np.array(self.scale_obs(obs_env), dtype=np.float32)
        done = self.check_done()
        rew = self.calc_reward()
        info = {}
        self.episode_step += 1
        return obs_agent, rew, done, info
    
    def check_done(self): 
        if abs(self.dQ_out) < self.dQ_threshold or self.N_steps_max < self.episode_step: 
            return True 
        else: 
            return False

    def calc_eff(self): 
        P_loss_gens, P_loss_grid = self.calc_losses()
        self.eff = (abs(self.P_out) + P_loss_gens) / (abs(self.P_out) + P_loss_gens + P_loss_grid)
        return self.eff 
    
    def calc_losses(self):
        P_loss_grid_mw = sum(self.P_gs_mva) - self.P_out 
        P_loss_gens = 0 
        for G, P, Q, V in zip(self.gens, self.P_gs_mva, self.Q_gs_mva, self.V_gs): 
            n, P_l_s, P_l_r, P_l_c = G.calc_losses_pu(P/G.md.Sn_mva, Q/G.md.Sn_mva, V)
            P_loss_gens += (P_l_s + P_l_r + P_l_c)*G.md.Sn_mva
        return P_loss_gens, P_loss_grid_mw
    
    def calc_reward(self):
        # cd_violation = [not CD.is_inside(P_g/Sn, Q_g/Sn) for CD, V_g, P_g, Q_g, Sn in zip(self.CDs, self.V_gs, self.P_gs_mva, self.Q_gs_mva, self.Sn)]
        cd_violation = np.array([False])
        V_out_reward = self.V_out_reward()
        if any(cd_violation) or abs(self.dQ_out) > self.dQ_threshold: 
            if self.episode_step < self.N_steps_max: 
                return self.step_reward
            else: 
                return self.failure
        else: 
            eff = self.calc_eff()
            return self.eff_reward(eff) + V_out_reward
        
class InitGrid: 
    def __init__(self): 
        self.S_sum = (G1.md.Sn_mva + G2.md.Sn_mva + G3.md.Sn_mva + G4.md.Sn_mva + G5.md.Sn_mva)
        self.P_min = 0.1 * self.S_sum
        self.P_max = 0.95 * self.S_sum
        self.V_mean = 1.0 
        self.V_std = 0.01 
        self.share = PSharing(np.zeros(5), np.array([G1.md.Sn_mva, G2.md.Sn_mva, G3.md.Sn_mva, G4.md.Sn_mva, G5.md.Sn_mva]))
        
    def find_Q_lim(self, P_gs, V_gs, V_ext, env): 
        env.set_net(P_gs, V_gs, V_ext, 0)
        env.get_obs()
        return env.Q_out
    
    def get_grid_data(self, env):
        P_tot = np.random.uniform(self.P_min, self.P_max) 
        V_ext = np.random.normal(self.V_mean, self.V_std) 
        
        P_gs = get_Pgs_mva(P_tot, env.Sn, self.share)
        Q_tot_min = self.find_Q_lim(P_gs, np.ones(5)*0.9, V_ext, env)
        Q_tot_max = self.find_Q_lim(P_gs, np.ones(5)*1.05, V_ext, env)
        Q_tot = np.random.uniform(Q_tot_min, Q_tot_max) 
        return P_tot, Q_tot, V_ext, P_gs

def get_Pgs_mva(P_tot_mva, Sn, share: PSharing):
    P_share = share.get_cP_curve(P_tot_mva) 
    inside = False
    while not inside:
        cP = np.random.random(len(Sn) - 1)
        P_g = P_share(cP)
        if not np.any(P_g < 0.0) and not np.any(P_g > Sn): 
            inside = True
    return P_g 