import SynGenLoss_v2 as sgl  
from SynGenLoss_v2.Model1 import GenDataClass, GeneratorModel, TrafoDataClass, TrafoModel, LineDataClass, LineModel
from SynGenLoss_v2.Model1.components.GenSaturationModel_v1 import SaturationModel

# Create G1
G1_data = GenDataClass() 
G1_data.standard_params(Sn_mva=103, V_nom_kV=11.0, cos_phi=0.9, If_nom_A=525.15, Ra=0.00182, Xd=1.059, Xq=0.676, Xp=0.141)
G1_data.nominal_losses(V_nom=1.0, Ia_nom=5406.1, If_nom=1065, P_an_kW=186.46, P_sn_kW=89.16, P_fn_kW=173.65, P_brn_kW=2.13, P_exn_kW=15.88, 
                          P_cn_kW=211.92, P_wfn_kW=172.92, P_bn_kW=240.90)
G1_sat = SaturationModel(bv=1.0, k=1.0308, Cm=0.160, m=7)
G1 = GeneratorModel(G1_data, G1_sat)

# Create G2 - Rad 33 i excel
G2_data = GenDataClass() 
G2_data.standard_params(Sn_mva=20, V_nom_kV=6.6, cos_phi=0.85, If_nom_A=356.3, Ra=0.0036, Xd=1.12, Xq=0.78, Xp=0.141)
G2_data.nominal_losses(V_nom=1.0, Ia_nom=1749.5, If_nom=712.7, P_an_kW=34.72, P_sn_kW=0, P_fn_kW=22.06, P_brn_kW=0, P_exn_kW=1.99, 
                          P_cn_kW=26.60, P_wfn_kW=401.15, P_bn_kW=0)
G2_sat = SaturationModel(bv=1.0, k=1.0308, Cm=0.160, m=7)
G2 = GeneratorModel(G2_data, G2_sat)

# Create G3 - Rad 36 i excel
G3_data = GenDataClass() 
G3_data.standard_params(Sn_mva=21, V_nom_kV=10.2, cos_phi=0.95, If_nom_A=170.9, Ra=0.0035, Xd=1.04, Xq=0.69, Xp=0.141)
G3_data.nominal_losses(V_nom=1.0, Ia_nom=1188.7, If_nom=341.7, P_an_kW=37.05, P_sn_kW=0, P_fn_kW=23.54, P_brn_kW=0, P_exn_kW=2.13, 
                          P_cn_kW=28.38, P_wfn_kW=286.9, P_bn_kW=0)
G3_sat = SaturationModel(bv=1.0, k=1.0308, Cm=0.160, m=7)
G3 = GeneratorModel(G3_data, G3_sat)

# Create G4 - Rad 87 i excel
G4_data = GenDataClass() 
G4_data.standard_params(Sn_mva=60, V_nom_kV=9.5, cos_phi=0.92, If_nom_A=455.8, Ra=0.0023, Xd=1.11, Xq=0.66, Xp=0.141)
G4_data.nominal_losses(V_nom=1.0, Ia_nom=3646.4, If_nom=911.5, P_an_kW=142.35, P_sn_kW=0, P_fn_kW=90.46, P_brn_kW=0, P_exn_kW=8.17, 
                          P_cn_kW=109.06, P_wfn_kW=459.97, P_bn_kW=0)
G4_sat = SaturationModel(bv=1.0, k=1.0308, Cm=0.160, m=7)
G4 = GeneratorModel(G4_data, G4_sat)

# Create G5 - Rad 100 i excel
G5_data = GenDataClass() 
G5_data.standard_params(Sn_mva=72, V_nom_kV=10.5, cos_phi=0.86, If_nom_A=320.2, Ra=0.002, Xd=1.08, Xq=0.70, Xp=0.141)
G5_data.nominal_losses(V_nom=1.0, Ia_nom=3849.0, If_nom=640.3, P_an_kW=175.37, P_sn_kW=0, P_fn_kW=111.44, P_brn_kW=0, P_exn_kW=10.07, 
                          P_cn_kW=134.35, P_wfn_kW=463.36, P_bn_kW=0)
G5_sat = SaturationModel(bv=1.0, k=1.0308, Cm=0.160, m=7)
G5 = GeneratorModel(G5_data, G5_sat)

# Transformers 
T1_data = TrafoDataClass(G1.md.Sn_mva, 132, G1.md.V_nom_kV, 0.11, 0.0031, 0.003, 0.001)
T2_data = TrafoDataClass(G2.md.Sn_mva, 132, G2.md.V_nom_kV, 0.11, 0.0031, 0.003, 0.001)
T3_data = TrafoDataClass(G3.md.Sn_mva, 132, G3.md.V_nom_kV, 0.11, 0.0031, 0.003, 0.001)
T4_data = TrafoDataClass(G4.md.Sn_mva, 132, G4.md.V_nom_kV, 0.11, 0.0031, 0.003, 0.001)
T5_data = TrafoDataClass(G5.md.Sn_mva, 132, G5.md.V_nom_kV, 0.11, 0.0031, 0.003, 0.001)

T1 = TrafoModel(T1_data)
T2 = TrafoModel(T2_data)
T3 = TrafoModel(T3_data)
T4 = TrafoModel(T4_data)
T5 = TrafoModel(T5_data)

L1_data = LineDataClass()
L1_data.define_params(100, 132, 0.00719, 0.35, 0.33, 50)
L1 = LineModel(L1_data)

L2_data = LineDataClass()
L2_data.define_params(100, 132, 0.00719, 0.35, 0.33, 7)
L2 = LineModel(L2_data)

L3_data = LineDataClass()
L3_data.define_params(100, 132, 0.00719, 0.35, 0.33, 40)
L3 = LineModel(L3_data)

L4_data = LineDataClass()
L4_data.define_params(100, 132, 0.00719, 0.35, 0.33, 14)
L4 = LineModel(L4_data)

L5_data = LineDataClass()
L5_data.define_params(100, 132, 0.00719, 0.35, 0.33, 2)
L5 = LineModel(L5_data)

L6_data = LineDataClass()
L6_data.define_params(100, 132, 0.00719, 0.35, 0.33, 3)
L6 = LineModel(L6_data)

L7_data = LineDataClass()
L7_data.define_params(100, 132, 0.00719, 0.35, 0.33, 30)
L7 = LineModel(L7_data)