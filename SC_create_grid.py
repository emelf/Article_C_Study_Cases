import pandapower as pp
from math import pi
S_gs = [103, 20, 21, 60, 72] # MVA 
V_gs = [11, 11, 11, 11, 11] # kV
V_inf = 132 #kV

r_lines = 0.00719 #ohm/km
x_lines = 0.35 #ohm/km
b_lines = 3.3 #uS/km
line_lengths = [50, 7, 40, 14, 2, 3, 30]
f_grid = 50 #Hz

trafo_dataset = [{"sn_mva": Sn,
                  "vn_hv_kv": V_inf,
                  "vn_lv_kv": Vg,
                  "vk_percent": 11,
                  "vkr_percent": 0.31,
                  "pfe_kw": 0.001*Sn*1000,
                  "i0_percent": 0.3,
                  "shift_degree": 0,
                  "tap_side": "lv",
                  "tap_neutral": 0,
                  "tap_min": -2,
                  "tap_max": 2,
                  "tap_step_degree": 0,
                  "tap_step_percent": 2.5,
                  "tap_phase_shifter": False,
                  "vk0_percent": 1,
                  "vkr0_percent": 0.78125,
                  "mag0_percent": 100,
                  "mag0_rx": 0.,
                  "si0_hv_partial": 0.9,} for Sn, Vg in zip(S_gs, V_gs)]

def create_grid(V_inf_pu: float, P_pu=0.9, V_g=1.0, min_vm_pu_bus=0.9, max_vm_pu_bus=1.1) -> dict: 
        """Creates a SMIB system for study cases. \n 
        returns net, Sn [MVA]"""
        net = pp.create_empty_network(name="Study case grid", f_hz=f_grid)
        B1 = pp.create_bus(net, vn_kv=V_inf, name="B1", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B2 = pp.create_bus(net, vn_kv=V_inf, name="B2", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B3 = pp.create_bus(net, vn_kv=V_inf, name="B3", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B4 = pp.create_bus(net, vn_kv=V_inf, name="B4", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B5 = pp.create_bus(net, vn_kv=V_inf, name="B5", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B6 = pp.create_bus(net, vn_kv=V_inf, name="B6", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B7 = pp.create_bus(net, vn_kv=V_inf, name="B7", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B8 = pp.create_bus(net, vn_kv=V_inf, name="B8", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B9 = pp.create_bus(net, vn_kv=V_gs[0], name="B9", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B10 = pp.create_bus(net, vn_kv=V_gs[1], name="B10", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B11 = pp.create_bus(net, vn_kv=V_gs[2], name="B11", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B12 = pp.create_bus(net, vn_kv=V_gs[3], name="B12", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        B13 = pp.create_bus(net, vn_kv=V_gs[4], name="B13", type="b", min_vm_pu=min_vm_pu_bus, max_vm_pu=max_vm_pu_bus)
        
        gens = [pp.create_gen(net, bus, p_mw=P_pu*Sn, vm_pu=V_g, sn_mva=Sn, name=f"G{i}", min_q_mvar=-Sn, max_q_mvar=Sn, min_p_mw=0, 
                              max_p_mw=Sn, controllable=True) for i, (bus, Sn) in enumerate(zip([B9, B10, B11, B12, B13], S_gs))]
        
        [pp.create_std_type(net, trafo_data, f"Plant_Trafo_{i+1}", element='trafo') for i, trafo_data in enumerate(trafo_dataset)]
        pp.create_transformer(net, B2, B9, "Plant_Trafo_1", name="T1", max_loading_percent=110)
        pp.create_transformer(net, B5, B10, "Plant_Trafo_2", name="T2", max_loading_percent=110)
        pp.create_transformer(net, B6, B11, "Plant_Trafo_3", name="T3", max_loading_percent=110)
        pp.create_transformer(net, B7, B12, "Plant_Trafo_4", name="T4", max_loading_percent=110)
        pp.create_transformer(net, B8, B13, "Plant_Trafo_5", name="T5", max_loading_percent=110)
        
        pp.create_line_from_parameters(net, B1, B2, length_km=line_lengths[0], r_ohm_per_km=r_lines, x_ohm_per_km=x_lines, 
                                       c_nf_per_km=b_lines*1000/(2*pi*f_grid), max_i_ka=2, name="L1", max_loading_percent=100)
        pp.create_line_from_parameters(net, B5, B2, length_km=line_lengths[1], r_ohm_per_km=r_lines, x_ohm_per_km=x_lines, 
                                       c_nf_per_km=b_lines*1000/(2*pi*f_grid), max_i_ka=2, name="L2", max_loading_percent=100)
        pp.create_line_from_parameters(net, B3, B2, length_km=line_lengths[2], r_ohm_per_km=r_lines, x_ohm_per_km=x_lines, 
                                       c_nf_per_km=b_lines*1000/(2*pi*f_grid), max_i_ka=2, name="L3", max_loading_percent=100)
        pp.create_line_from_parameters(net, B6, B3, length_km=line_lengths[3], r_ohm_per_km=r_lines, x_ohm_per_km=x_lines, 
                                       c_nf_per_km=b_lines*1000/(2*pi*f_grid), max_i_ka=2, name="L4", max_loading_percent=100)
        pp.create_line_from_parameters(net, B3, B4, length_km=line_lengths[4], r_ohm_per_km=r_lines, x_ohm_per_km=x_lines, 
                                       c_nf_per_km=b_lines*1000/(2*pi*f_grid), max_i_ka=2, name="L5", max_loading_percent=100)
        pp.create_line_from_parameters(net, B7, B4, length_km=line_lengths[5], r_ohm_per_km=r_lines, x_ohm_per_km=x_lines, 
                                       c_nf_per_km=b_lines*1000/(2*pi*f_grid), max_i_ka=2, name="L6", max_loading_percent=100)
        pp.create_line_from_parameters(net, B8, B4, length_km=line_lengths[6], r_ohm_per_km=r_lines, x_ohm_per_km=x_lines, 
                                       c_nf_per_km=b_lines*1000/(2*pi*f_grid), max_i_ka=2, name="L6", max_loading_percent=100)
        
        pp.create_ext_grid(net, B1, vm_pu=V_inf_pu, name="Ext Grid")
        return net