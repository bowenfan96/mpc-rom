import csv
import datetime
import pickle

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyomo.environ import *
from pyomo.dae import *
from pyomo.solvers import *

from heatEq_nn_controller import *

results_folder = "expReplay_results/edge09/"


class HeatEqSimulator:
    def __init__(self, duration=1, N=20):
        # Unique ID for savings csv and plots, based on model timestamp
        self.uid = datetime.datetime.now().strftime("%d-%H.%M.%S.%f")[:-3]
        print("Model uid", self.uid)

        # ----- MODEL MATRICES AS FITTED BY SINDY -----
        # self.A = [
        #     [-291.739, 1790.564, -3101.847, -960.982, 8809.239, -9848.806, 4125.694, -592.540, -1680.628, 4317.733,
        #      -3546.105, 3803.179, -8000.066, 5568.549, 5128.455, -8188.630, -155.974, 5473.816, -3395.253, 669.538],
        #     [-80.417, 428.846, 10.318, -3332.862, 7123.322, -6216.818, 3234.441, -1799.858, -1375.549, 4037.545,
        #      -1821.827, 1647.447, -6374.643, 4534.067, 5277.673, -7695.563, -230.501, 4980.652, -3009.366, 580.213],
        #     [289.324, -1923.792, 5414.758, -7726.019, 4958.247, -801.749, 2368.327, -4523.979, -512.232, 4052.072,
        #      69.932, -838.998, -4793.367, 3398.342, 5710.560, -6743.409, -1562.855, 5540.795, -3027.964, 551.419],
        #     [693.896, -4471.120, 11194.267, -12305.677, 2505.704, 4952.475, 1592.310, -7589.851, 619.599, 3991.135,
        #      1790.046, -3137.400, -3064.336, 1983.822, 6041.307, -5352.769, -3227.641, 6200.421, -3053.968, 521.699],
        #     [988.854, -6288.547, 15149.575, -15035.259, 119.646, 9295.862, 993.262, -9655.866, 1696.066, 3465.806,
        #      2896.806, -4582.882, -1098.241, 209.915, 5786.960, -3484.389, -4335.161, 6038.285, -2709.488, 433.850],
        #     [1060.945, -6658.380, 15614.667, -14485.442, -1704.164, 10733.337, 680.629, -9712.480, 2427.012, 2283.197,
        #      3010.787, -4627.869, 986.886, -1818.068, 4633.672, -1176.570, -4358.090, 4574.282, -1815.843, 263.528],
        #     [838.416, -5136.509, 11602.711, -9912.783, -2493.584, 8291.894, 697.050, -7139.646, 2573.107, 454.687,
        #      1864.063, -2913.944, 2914.277, -3835.260, 2403.152, 1585.786, -3286.703, 1856.341, -415.921, 20.995],
        #     [371.285, -2055.366, 3977.218, -2312.351, -2015.765, 2421.696, 1024.551, -2427.562, 2108.710, -1606.694,
        #      -479.563, 373.400, 4335.084, -5583.149, -580.092, 4826.550, -2045.190, -1130.222, 1066.811, -226.125],
        #     [-210.653, 1749.546, -5224.888, 6287.414, -337.992, -5392.391, 1550.793, 3244.963, 1146.961, -3235.042,
        #      -3789.151, 4734.851, 4948.529, -6952.640, -3683.110, 8759.436, -2324.942, -2736.369, 1957.766, -374.196],
        #     [-702.758, 4659.027, -11618.354, 11549.555, 1290.147, -10072.760, 1100.396, 7374.783, -192.926, -4563.562,
        #      -2647.122, 4479.550, 2896.549, -2651.967, -5021.093, 3424.474, 5547.555, -7590.735, 3429.781, -552.421],
        #     [-933.772, 6003.769, -14346.637, 13105.144, 3274.052, -12776.131, 758.978, 9383.972, -1420.686, -3851.390,
        #      -3080.154, 5143.823, 575.420, 0, -5202.984, 2313.365, 5046.615, -6002.242, 2514.186, -379.398],
        #     [-795.439, 4771.733, -10253.173, 6827.778, 7045.490, -11930.671, -323.399, 8660.273, -2679.439, -1222.561,
        #      -2508.731, 3739.421, -2461.029, 4313.689, -5156.274, 527.084, 4105.629, -3813.978, 1468.909, -223.343],
        #     [-349.280, 1895.309, -3319.727, 395.159, 5363.429, -5121.805, -1280.926, 4329.701, -2712.501, 1500.948,
        #      -628.664, 920.865, -4248.296, 5110.458, -607.761, -1557.494, -1609.404, 3408.088, -1837.479, 331.674],
        #     [169.803, -1429.744, 4596.862, -6795.281, 3408.056, 2296.599, -2234.798, -327.555, -2399.240, 3921.074,
        #      1116.083, -1941.959, -4787.251, 4739.148, 3355.341, -2828.136, -6371.269, 8945.647, -4268.884, 727.539],
        #     [582.732, -4030.018, 10615.497, -11898.386, 1363.635, 8114.333, -2899.725, -3805.793, -1865.448, 5295.363,
        #      2101.902, -3827.250, -3893.729, 2970.390, 5959.444, -2951.115, -9173.673, 11635.243, -5345.027, 894.170],
        #     [795.411, -5306.610, 13327.306, -13644.656, -493.859, 11191.115, -3156.298, -5399.765, -1237.959, 5399.660,
        #      2090.297, -4278.487, -1893.340, 154.542, 6929.972, -2023.357, -9726.266, 11321.194, -5044.689, 834.731],
        #     [0, -598.538, 1902.237, -6770.018, 0, 8623.605, -477.506, -4378.292, -812.180, 3081.153, 1424.403,
        #      -2894.482, 1918.617, -5273.844, 6710.398, 1127.968, -9452.731, 8704.798, -3526.167, 560.197],
        #     [793.445, -5089.628, 12016.326, -10656.858, -3305.893, 11674.740, -3038.019, -4768.283, 75.726, 3424.876,
        #      -8.929, -2076.187, 3145.950, -6630.769, 6172.227, 1321.311, -7275.225, 6166.778, -2427.773, 393.243],
        #     [735.011, -4622.565, 10549.740, -8532.701, -4319.231, 11057.774, -2872.967, -4111.633, 924.342, 2147.955,
        #      -1445.299, -285.132, 5294.751, -9924.299, 5732.956, 3027.372, -6064.608, 3616.956, -1139.782, 176.990],
        #     [778.998, -4828.564, 10778.963, -8227.060, -5144.944, 11529.224, -2785.120, -4538.974, 2074.409, 1281.485,
        #      -2650.771, 1280.107, 6732.516, -12956.954, 6342.055, 4368.772, -6383.942, 2865.995, -640.287, 86.459]]
        # self.A = np.array(self.A)
        #
        # self.B = [[4.851, 0],
        #           [4.648, 0],
        #           [4.337, 0],
        #           [3.918, 0],
        #           [3.379, 0],
        #           [2.790, 0],
        #           [2.163, 0],
        #           [1.587, 0],
        #           [1.093, 0],
        #           [0.709, 0.404],
        #           [0.421, 0.725],
        #           [0, 1.171],
        #           [0, 1.698],
        #           [0, 2.282],
        #           [0, 2.862],
        #           [0, 3.362],
        #           [0, 3.848],
        #           [0, 4.294],
        #           [0, 4.632],
        #           [0, 4.874]]
        #
        # self.B = np.array(self.B)

        self.A = [[1316.423,218.680,-3901.499,4476.847,-5040.135,-2111.942,10532.186,484.539,-7873.884,-5422.477,6532.657,7150.517,-6223.806,-1145.742,-1319.583,4700.185,-3854.379,1387.137,1363.642,-1175.477],
[578.346,8.650,-1515.445,1721.207,-1760.871,-1098.072,4049.051,159.442,-2674.971,-1963.798,2219.271,2295.012,-2151.018,104.671,-765.076,1574.180,-1353.458,620.370,316.098,-331.685],
[-30.670,-210.532,460.662,-431.293,1130.964,-1223.415,0,-796.093,1620.378,937.546,-1497.185,-1368.808,1378.376,492.305,-51.732,-959.495,884.832,-316.759,-335.088,291.537],
[-375.208,-408.008,1580.334,-1438.506,3106.510,-2995.504,0,-2540.515,4044.542,2681.410,-3858.712,-2889.792,3661.349,-315.078,704.480,-2361.161,2751.762,-1835.706,0,423.002],
[-449.839,-393.391,1725.204,-1539.269,3079.395,-2844.985,0,-2656.266,3909.298,2666.839,-3804.173,-2441.999,3645.242,-1146.207,1152.171,-2155.534,2467.207,-1598.122,-78.412,395.331],
[-326.349,-63.317,1038.194,-1167.664,944.171,1037.677,-2692.580,43.326,1393.917,1063.429,-1222.730,-886.443,1209.234,-779.859,752.139,-704.309,751.635,-558.999,91.604,51.500],
[-107.226,1414.092,-3941.821,5870.186,-7251.994,7722.948,-6530.022,5254.181,-845.676,-8797.247,13311.059,-8062.621,2064.174,1870.653,-3111.729,564.128,0,2259.589,-1899.221,314.453],
[987.155,0,-5431.838,11666.327,-16686.044,14923.090,-5611.809,0,5260.233,-18330.567,24791.768,-16743.130,5047.479,5663.884,-7003.928,-2965.187,6617.377,-1018.844,-1010.187,0],
[-86.808,5746.112,-17059.553,24581.243,-27725.745,26856.941,-23030.007,19596.045,-1875.672,-33413.190,52103.702,-37554.016,9404.457,16341.804,-19214.056,0,8246.485,0,-3284.925,783.053],
[247.392,148.493,-1045.612,1195.227,-951.598,-1018.647,2678.482,-31.146,-1461.486,-1106.686,1251.875,965.797,-1195.844,711.116,-736.136,730.004,-761.772,546.794,-128.946,-12.456],
[-14.463,23.466,-88.867,39.174,693.875,-987.389,-58.159,-226.039,1216.700,557.367,-954.720,-1777.138,815.895,2080.431,-815.025,-1002.888,492.576,350.845,-852.408,504.284],
[-3675.625,9301.381,-13424.674,16656.871,-3922.887,-21352.028,22675.153,-905.509,-7815.743,1292.576,2107.158,-4179.663,14749.323,-28251.969,25760.487,-11611.879,14550.761,-30968.122,27819.034,-8819.168],
[-2139.659,4803.385,-6247.482,7673.199,-822.714,-10789.307,9509.445,-562.426,-2134.698,2058.554,-1018.794,-4019.715,10362.735,-14093.181,9091.343,0,2997.754,-14260.788,13898.206,-4388.051],
[-348.444,0,-42.783,2022.685,-372.534,-4379.857,3971.374,-230.475,-1127.340,899.804,-108.844,-1400.785,4273.501,-7808.297,7041.791,-3219.688,4034.777,-8116.474,7241.139,-2348.161],
[61.063,-104.309,166.991,-131.925,-655.186,1065.785,-115.748,309.497,-1109.303,-559.027,816.395,1808.935,-718.031,-2249.658,849.189,1045.034,-446.446,-424.988,995.232,-604.838],
[291.612,-51.533,-587.237,872.061,-2098.178,1120.339,2075.383,434.434,-3268.688,-1905.966,2766.892,3950.039,-2746.582,-3122.911,2162.597,82.451,0,0,788.334,-724.695],
[333.820,47.962,-928.313,1185.301,-2172.423,571.491,2914.204,392.101,-3542.459,-2145.927,2858.054,4050.010,-2562.813,-2832.673,532.949,2545.210,-1635.432,-26.140,1317.575,-876.582],
[160.426,78.511,-612.297,745.239,-928.389,-190.276,1715.865,103.716,-1499.664,-987.558,1228.566,1484.109,-1096.714,-564.387,-116.748,964.395,-721.780,180.576,237.706,-168.692],
[-204.988,69.966,279.034,-417.851,1489.095,-1105.673,-1235.179,-347.405,2440.758,1332.753,-1958.013,-3147.691,1769.954,2986.868,-935.569,-1949.173,1085.275,322.681,-1472.847,980.355],
[-605.406,-192.170,1433.578,-1432.719,5454.619,-6240.556,0,-3608.674,7840.431,4613.900,-6902.567,-8218.354,6301.160,5600.673,-1269.047,-5702.860,4180.380,-566.287,-3030.148,2265.607]]

        self.A = np.array(self.A)
        
        self.B = [[4.701,0],
[4.247, 0],
[3.713 ,0.001],
[3.120, 0.003],
[2.527, 0.007],
[1.960, 0.015],
[1.439, 0],
[1.009, 0],
[0.665, 0],
[0.402, 0.234],
[0.235, 0.404],
[0.685, 0],
[1.053, 0],
[1.476, 0],
[0.015, 1.995],
[0.007, 2.590],
[0.003, 3.190],
[0.001, 3.744],
[4.252, 0],
[4.711 ,0]]

        self.B = np.array(self.B)

        # ----- SET UP THE BASIC MODEL -----
        # Set up pyomo model structure
        self.model = ConcreteModel()
        self.model.time = ContinuousSet(bounds=(0, duration))

        # Initial state: the rod is 273 Kelvins throughout
        # Change this array if random initial states are desired
        x_init = np.full(shape=(N,), fill_value=273)
        # x_init = np.random.randint(low=263, high=283, size=N)

        # NOTE: Pyomo can simulate via scipy/casadi only if:
        # 1. model.u is indexed only by time, so Bu using matrix multiplication is not possible
        # 2. model contains if statements, so the ode cannot have conditions
        # After trying many things, this (silly) method seems to be the only way
        # if we want pyomo to simulate with random controls

        # Set up all the finite elements
        self.model.x0 = Var(self.model.time)
        self.model.x0_dot = DerivativeVar(self.model.x0, wrt=self.model.time)
        self.model.x0[0].fix(x_init[0])
        self.model.x1 = Var(self.model.time)
        self.model.x1_dot = DerivativeVar(self.model.x1, wrt=self.model.time)
        self.model.x1[0].fix(x_init[1])
        self.model.x2 = Var(self.model.time)
        self.model.x2_dot = DerivativeVar(self.model.x2, wrt=self.model.time)
        self.model.x2[0].fix(x_init[2])
        self.model.x3 = Var(self.model.time)
        self.model.x3_dot = DerivativeVar(self.model.x3, wrt=self.model.time)
        self.model.x3[0].fix(x_init[3])
        self.model.x4 = Var(self.model.time)
        self.model.x4_dot = DerivativeVar(self.model.x4, wrt=self.model.time)
        self.model.x4[0].fix(x_init[4])
        self.model.x5 = Var(self.model.time)
        self.model.x5_dot = DerivativeVar(self.model.x5, wrt=self.model.time)
        self.model.x5[0].fix(x_init[5])
        self.model.x6 = Var(self.model.time)
        self.model.x6_dot = DerivativeVar(self.model.x6, wrt=self.model.time)
        self.model.x6[0].fix(x_init[6])
        self.model.x7 = Var(self.model.time)
        self.model.x7_dot = DerivativeVar(self.model.x7, wrt=self.model.time)
        self.model.x7[0].fix(x_init[7])
        self.model.x8 = Var(self.model.time)
        self.model.x8_dot = DerivativeVar(self.model.x8, wrt=self.model.time)
        self.model.x8[0].fix(x_init[8])
        self.model.x9 = Var(self.model.time)
        self.model.x9_dot = DerivativeVar(self.model.x9, wrt=self.model.time)
        self.model.x9[0].fix(x_init[9])
        self.model.x10 = Var(self.model.time)
        self.model.x10_dot = DerivativeVar(self.model.x10, wrt=self.model.time)
        self.model.x10[0].fix(x_init[10])
        self.model.x11 = Var(self.model.time)
        self.model.x11_dot = DerivativeVar(self.model.x11, wrt=self.model.time)
        self.model.x11[0].fix(x_init[11])
        self.model.x12 = Var(self.model.time)
        self.model.x12_dot = DerivativeVar(self.model.x12, wrt=self.model.time)
        self.model.x12[0].fix(x_init[12])
        self.model.x13 = Var(self.model.time)
        self.model.x13_dot = DerivativeVar(self.model.x13, wrt=self.model.time)
        self.model.x13[0].fix(x_init[13])
        self.model.x14 = Var(self.model.time)
        self.model.x14_dot = DerivativeVar(self.model.x14, wrt=self.model.time)
        self.model.x14[0].fix(x_init[14])
        self.model.x15 = Var(self.model.time)
        self.model.x15_dot = DerivativeVar(self.model.x15, wrt=self.model.time)
        self.model.x15[0].fix(x_init[15])
        self.model.x16 = Var(self.model.time)
        self.model.x16_dot = DerivativeVar(self.model.x16, wrt=self.model.time)
        self.model.x16[0].fix(x_init[16])
        self.model.x17 = Var(self.model.time)
        self.model.x17_dot = DerivativeVar(self.model.x17, wrt=self.model.time)
        self.model.x17[0].fix(x_init[17])
        self.model.x18 = Var(self.model.time)
        self.model.x18_dot = DerivativeVar(self.model.x18, wrt=self.model.time)
        self.model.x18[0].fix(x_init[18])
        self.model.x19 = Var(self.model.time)
        self.model.x19_dot = DerivativeVar(self.model.x19, wrt=self.model.time)
        self.model.x19[0].fix(x_init[19])

        # Set up controls
        self.model.u0 = Var(self.model.time, bounds=(173, 473))
        self.model.u1 = Var(self.model.time, bounds=(173, 473))

        sigma = -5.67e-8 / 2
        env_temp = 273

        # ODEs
        def _ode_x0(m, _t):
            return m.x0_dot[_t] == self.A[0][0] * m.x0[_t] + self.A[0][1] * m.x1[_t] + self.B[0][0] * m.u0[
                _t] + sigma * (m.x0[_t] ** 4 - env_temp ** 4)

        self.model.x0_ode = Constraint(self.model.time, rule=_ode_x0)

        # Set up x1_dot to x18_dot = Ax only
        def _ode_x1(m, _t):
            return m.x1_dot[_t] == self.A[1][1 - 1] * m.x0[_t] + self.A[1][1] * m.x1[_t] + self.A[1][1 + 1] * m.x2[
                _t] + sigma * (m.x1[_t] ** 4 - env_temp ** 4)

        self.model.x1_ode = Constraint(self.model.time, rule=_ode_x1)

        def _ode_x2(m, _t):
            return m.x2_dot[_t] == self.A[2][2 - 1] * m.x1[_t] + self.A[2][2] * m.x2[_t] + self.A[2][2 + 1] * m.x3[
                _t] + sigma * (m.x2[_t] ** 4 - env_temp ** 4)

        self.model.x2_ode = Constraint(self.model.time, rule=_ode_x2)

        def _ode_x3(m, _t):
            return m.x3_dot[_t] == self.A[3][3 - 1] * m.x2[_t] + self.A[3][3] * m.x3[_t] + self.A[3][3 + 1] * m.x4[
                _t] + sigma * (m.x3[_t] ** 4 - env_temp ** 4)

        self.model.x3_ode = Constraint(self.model.time, rule=_ode_x3)

        def _ode_x4(m, _t):
            return m.x4_dot[_t] == self.A[4][4 - 1] * m.x3[_t] + self.A[4][4] * m.x4[_t] + self.A[4][4 + 1] * m.x5[
                _t] + sigma * (m.x4[_t] ** 4 - env_temp ** 4)

        self.model.x4_ode = Constraint(self.model.time, rule=_ode_x4)

        def _ode_x5(m, _t):
            return m.x5_dot[_t] == self.A[5][5 - 1] * m.x4[_t] + self.A[5][5] * m.x5[_t] + self.A[5][5 + 1] * m.x6[
                _t] + sigma * (m.x5[_t] ** 4 - env_temp ** 4)

        self.model.x5_ode = Constraint(self.model.time, rule=_ode_x5)

        def _ode_x6(m, _t):
            return m.x6_dot[_t] == self.A[6][6 - 1] * m.x5[_t] + self.A[6][6] * m.x6[_t] + self.A[6][6 + 1] * m.x7[
                _t] + sigma * (m.x6[_t] ** 4 - env_temp ** 4)

        self.model.x6_ode = Constraint(self.model.time, rule=_ode_x6)

        def _ode_x7(m, _t):
            return m.x7_dot[_t] == self.A[7][7 - 1] * m.x6[_t] + self.A[7][7] * m.x7[_t] + self.A[7][7 + 1] * m.x8[
                _t] + sigma * (m.x7[_t] ** 4 - env_temp ** 4)

        self.model.x7_ode = Constraint(self.model.time, rule=_ode_x7)

        def _ode_x8(m, _t):
            return m.x8_dot[_t] == self.A[8][8 - 1] * m.x7[_t] + self.A[8][8] * m.x8[_t] + self.A[8][8 + 1] * m.x9[
                _t] + sigma * (m.x8[_t] ** 4 - env_temp ** 4)

        self.model.x8_ode = Constraint(self.model.time, rule=_ode_x8)

        def _ode_x9(m, _t):
            return m.x9_dot[_t] == self.A[9][9 - 1] * m.x8[_t] + self.A[9][9] * m.x9[_t] + self.A[9][9 + 1] * m.x10[
                _t] + sigma * (m.x9[_t] ** 4 - env_temp ** 4)

        self.model.x9_ode = Constraint(self.model.time, rule=_ode_x9)

        def _ode_x10(m, _t):
            return m.x10_dot[_t] == self.A[10][10 - 1] * m.x9[_t] + self.A[10][10] * m.x10[_t] + self.A[10][10 + 1] * \
                   m.x11[_t] + sigma * (m.x10[_t] ** 4 - env_temp ** 4)

        self.model.x10_ode = Constraint(self.model.time, rule=_ode_x10)

        def _ode_x11(m, _t):
            return m.x11_dot[_t] == self.A[11][11 - 1] * m.x10[_t] + self.A[11][11] * m.x11[_t] + self.A[11][11 + 1] * \
                   m.x12[_t] + sigma * (m.x11[_t] ** 4 - env_temp ** 4)

        self.model.x11_ode = Constraint(self.model.time, rule=_ode_x11)

        def _ode_x12(m, _t):
            return m.x12_dot[_t] == self.A[12][12 - 1] * m.x11[_t] + self.A[12][12] * m.x12[_t] + self.A[12][12 + 1] * \
                   m.x13[_t] + sigma * (m.x12[_t] ** 4 - env_temp ** 4)

        self.model.x12_ode = Constraint(self.model.time, rule=_ode_x12)

        def _ode_x13(m, _t):
            return m.x13_dot[_t] == self.A[13][13 - 1] * m.x12[_t] + self.A[13][13] * m.x13[_t] + self.A[13][13 + 1] * \
                   m.x14[_t] + sigma * (m.x13[_t] ** 4 - env_temp ** 4)

        self.model.x13_ode = Constraint(self.model.time, rule=_ode_x13)

        def _ode_x14(m, _t):
            return m.x14_dot[_t] == self.A[14][14 - 1] * m.x13[_t] + self.A[14][14] * m.x14[_t] + self.A[14][14 + 1] * \
                   m.x15[_t] + sigma * (m.x14[_t] ** 4 - env_temp ** 4)

        self.model.x14_ode = Constraint(self.model.time, rule=_ode_x14)

        def _ode_x15(m, _t):
            return m.x15_dot[_t] == self.A[15][15 - 1] * m.x14[_t] + self.A[15][15] * m.x15[_t] + self.A[15][15 + 1] * \
                   m.x16[_t] + sigma * (m.x15[_t] ** 4 - env_temp ** 4)

        self.model.x15_ode = Constraint(self.model.time, rule=_ode_x15)

        def _ode_x16(m, _t):
            return m.x16_dot[_t] == self.A[16][16 - 1] * m.x15[_t] + self.A[16][16] * m.x16[_t] + self.A[16][16 + 1] * \
                   m.x17[_t] + sigma * (m.x16[_t] ** 4 - env_temp ** 4)

        self.model.x16_ode = Constraint(self.model.time, rule=_ode_x16)

        def _ode_x17(m, _t):
            return m.x17_dot[_t] == self.A[17][17 - 1] * m.x16[_t] + self.A[17][17] * m.x17[_t] + self.A[17][17 + 1] * \
                   m.x18[_t] + sigma * (m.x17[_t] ** 4 - env_temp ** 4)

        self.model.x17_ode = Constraint(self.model.time, rule=_ode_x17)

        def _ode_x18(m, _t):
            return m.x18_dot[_t] == self.A[18][18 - 1] * m.x17[_t] + self.A[18][18] * m.x18[_t] + self.A[18][18 + 1] * \
                   m.x19[_t] + sigma * (m.x18[_t] ** 4 - env_temp ** 4)

        self.model.x18_ode = Constraint(self.model.time, rule=_ode_x18)

        # Set up x19_dot = Ax + Bu
        def _ode_x19(m, _t):
            return m.x19_dot[_t] == self.A[19][19] * m.x19[_t] + self.A[19][18] * m.x18[_t] + self.B[19][1] * m.u1[
                _t] + sigma * (m.x19[_t] ** 4 - env_temp ** 4)

        self.model.x19_ode = Constraint(self.model.time, rule=_ode_x19)

        # Lagrangian cost
        self.model.L = Var(self.model.time)
        self.model.L_dot = DerivativeVar(self.model.L, wrt=self.model.time)
        self.model.L[0].fix(0)

        # ----- OBJECTIVE AND COST FUNCTION -----
        # Objective:
        # We want to heat element 6 (x[5]) at the 1/3 position to 30 C, 303 K
        # And element 14 (x[13]) at the 2/3 position to 60 C, 333 K
        # We would like to minimize the controller costs too, in terms of how much heating or cooling is applied

        # Define weights for setpoint and controller objectives
        setpoint_weight = 0.995
        controller_weight = 1 - setpoint_weight

        # Lagrangian cost
        def _Lagrangian(m, _t):
            return m.L_dot[_t] \
                   == setpoint_weight * ((m.x5[_t] - 303) ** 2 + (m.x13[_t] - 333) ** 2) \
                   + controller_weight * ((m.u0[_t] - 273) ** 2 + (m.u1[_t] - 273) ** 2)

        self.model.L_integral = Constraint(self.model.time, rule=_Lagrangian)

        # Objective function is to minimize the Lagrangian cost integral
        def _objective(m):
            return m.L[m.time.last()] - m.L[0]

        self.model.objective = Objective(rule=_objective, sense=minimize)

        # Constraint for the element at the 1/3 position: temperature must not exceed 313 K (10 K above setpoint)
        def _constraint_x5(m, _t):
            return m.x5[_t] <= 313

        self.model.constraint_x5 = Constraint(self.model.time, rule=_constraint_x5)

        # ----- DISCRETIZE THE MODEL INTO FINITE ELEMENTS -----
        # We need to discretize before adding ODEs in matrix form
        # We fix finite elements at 10, collocation points at 4, controls to be piecewise linear
        discretizer = TransformationFactory("dae.collocation")
        discretizer.apply_to(self.model, nfe=10, ncp=4, scheme="LAGRANGE-RADAU")

        # Make controls piecewise linear
        discretizer.reduce_collocation_points(self.model, var=self.model.u0, ncp=1, contset=self.model.time)
        discretizer.reduce_collocation_points(self.model, var=self.model.u1, ncp=1, contset=self.model.time)

        return

    def mpc_control(self):
        mpc_solver = SolverFactory("ipopt", tee=True)
        # mpc_solver.options['max_iter'] = 10000
        mpc_results = mpc_solver.solve(self.model)

        return mpc_results

    def parse_mpc_results(self):
        # Each t, x0, x1, x2, etc, U, L, instantaneous cost, cost to go, should be a column
        # Label them and return a pandas dataframe
        t = []
        # x is a list of lists
        x = []
        u0 = []
        u1 = []
        L = []
        inst_cost = []
        ctg = []

        # Record data at the intervals of finite elements only (0.1s), do not include collocation points
        timesteps = [timestep / 10 for timestep in range(11)]
        for time in self.model.time:
            if time in timesteps:
                t.append(time)
                u0.append(value(self.model.u0[time]))
                u1.append(value(self.model.u1[time]))
                L.append(value(self.model.L[time]))

                # Get all the x values
                temp_x = []
                temp_x.append(value(self.model.x0[time]))
                temp_x.append(value(self.model.x1[time]))
                temp_x.append(value(self.model.x2[time]))
                temp_x.append(value(self.model.x3[time]))
                temp_x.append(value(self.model.x4[time]))
                temp_x.append(value(self.model.x5[time]))
                temp_x.append(value(self.model.x6[time]))
                temp_x.append(value(self.model.x7[time]))
                temp_x.append(value(self.model.x8[time]))
                temp_x.append(value(self.model.x9[time]))
                temp_x.append(value(self.model.x10[time]))
                temp_x.append(value(self.model.x11[time]))
                temp_x.append(value(self.model.x12[time]))
                temp_x.append(value(self.model.x13[time]))
                temp_x.append(value(self.model.x14[time]))
                temp_x.append(value(self.model.x15[time]))
                temp_x.append(value(self.model.x16[time]))
                temp_x.append(value(self.model.x17[time]))
                temp_x.append(value(self.model.x18[time]))
                temp_x.append(value(self.model.x19[time]))
                x.append(temp_x)

        # Make sure all 11 time steps are recorded; this was problematic due to Pyomo's float indexing
        assert len(t) == 11

        for time in range(len(t)):
            # Instantaneous cost is L[t1] - L[t0]
            if time == 0:
                inst_cost.append(0)
            else:
                inst_cost.append(L[time] - L[time - 1])

        # Calculate cost to go
        for time in range(len(t)):
            ctg.append(inst_cost[time])
        # Sum backwards from tf
        for time in reversed(range(len(t) - 1)):
            ctg[time] += ctg[time + 1]

        x = np.array(x)

        df_data = {"t": t}
        for x_idx in range(self.A.shape[0]):
            df_data["x{}".format(x_idx)] = x[:, x_idx]
        df_data["u0"] = u0
        df_data["u1"] = u1
        df_data["L"] = L
        df_data["inst_cost"] = inst_cost
        df_data["ctg"] = ctg

        mpc_results_df = pd.DataFrame(df_data)
        mpc_results_df_dropped_t0 = mpc_results_df.drop(index=0)

        return mpc_results_df, mpc_results_df_dropped_t0

    def simulate_system_nn_controls(self):
        timesteps = [timestep / 10 for timestep in range(11)]
        u0_nn = np.full(shape=(20, ), fill_value=273)
        u1_nn = np.full(shape=(20, ), fill_value=273)

        self.model.var_input = Suffix(direction=Suffix.LOCAL)

        # Create a dictionary of piecewise linear controller actions
        u0_nn_profile = {timesteps[i]: u0_nn[i] for i in range(len(timesteps))}
        u1_nn_profile = {timesteps[i]: u1_nn[i] for i in range(len(timesteps))}

        # Update the control sequence to Pyomo
        self.model.var_input[self.model.u0] = u0_nn_profile
        self.model.var_input[self.model.u1] = u1_nn_profile

        sim = Simulator(self.model, package="casadi")
        tsim, profiles = sim.simulate(numpoints=11, varying_inputs=self.model.var_input, integrator='rk')

        print(profiles)

        # For some reason both tsim and profiles contain duplicates
        # Use pandas to drop the duplicates first
        # profiles columns: x0, x1, ..., x19, L
        temp_dict = {"t": tsim}
        for j in range(20):
            temp_dict["x{}".format(j)] = profiles[:, j]
        temp_dict["L"] = profiles[:, 20]

        deduplicate_df = pd.DataFrame(temp_dict)
        deduplicate_df = deduplicate_df.round(4)
        deduplicate_df.drop_duplicates(ignore_index=True, inplace=True)

        # Make dataframe from the simulator results
        t = deduplicate_df["t"]
        x = []
        for j in range(20):
            x.append(deduplicate_df["x{}".format(j)])

        # Note: at this point, x is a list of 20 pandas series, each series has 11 rows
        # Check duplicates were removed correctly
        assert len(t) == 11

        # Make dataframe from the final simulator results
        t = deduplicate_df["t"]
        x = []
        for i in range(20):
            x.append(deduplicate_df["x{}".format(i)])
        L = deduplicate_df["L"]
        u0 = u0_nn
        u1 = u1_nn
        inst_cost = []
        ctg = []

        for time in range(len(t)):
            # Instantaneous cost is L[t1] - L[t0]
            if time == 10:
                inst_cost.append(0)
            else:
                inst_cost.append(L[time + 1] - L[time])

        # Calculate cost to go
        for time in range(len(t)):
            ctg.append(inst_cost[time])
        # Sum backwards from tf
        for time in reversed(range(len(t) - 1)):
            ctg[time] += ctg[time + 1]

        # Calculate path violations
        path = [x[5][int(time * 10)] - 313 for time in t]
        path_violation = []
        for p in path:
            if max(path) > 0:
                path_violation.append(max(path))
            else:
                path_violation.append(p)

        temp_dict = {"t": t}
        for i in range(20):
            temp_dict["x{}".format(i)] = x[i]
        temp_dict["u0"] = u0
        temp_dict["u1"] = u1
        temp_dict["L"] = L
        temp_dict["inst_cost"] = inst_cost
        temp_dict["ctg"] = ctg
        temp_dict["path_diff"] = path_violation

        nn_sim_results_df = pd.DataFrame(temp_dict)
        nn_sim_results_df_dropped_tf = nn_sim_results_df.drop(index=10)

        return nn_sim_results_df, nn_sim_results_df_dropped_tf

    def simulate_system_sindy_controls(self):
        timesteps = [timestep / 10 for timestep in range(11)]

        u0_nn = [322.9563254,
                 173.0186607,
                 173.0188708,
                 472.8931624,
                 283.0565085,
                 417.8779231,
                 410.1953802,
                 464.3188755,
                 472.8905774,
                 472.8924449,
                 472.891716

                 ]
        u1_nn = [322.9702283,
                 472.7133161,
                 472.7132842,
                 472.7130589,
                 273.6717801,
                 290.5229226,
                 358.4480267,
                 388.3372644,
                 421.8174325,
                 460.1723944,
                 472.6993367

                 ]

        self.model.var_input = Suffix(direction=Suffix.LOCAL)
        # Create a dictionary of piecewise linear controller actions
        u0_nn_profile = {timesteps[i]: u0_nn[i] for i in range(len(timesteps))}
        u1_nn_profile = {timesteps[i]: u1_nn[i] for i in range(len(timesteps))}

        # Update the control sequence to Pyomo
        self.model.var_input[self.model.u0] = u0_nn_profile
        self.model.var_input[self.model.u1] = u1_nn_profile

        sim = Simulator(self.model, package="casadi")
        tsim, profiles = sim.simulate(numpoints=11, varying_inputs=self.model.var_input)

        # For some reason both tsim and profiles contain duplicates
        # Use pandas to drop the duplicates first
        # profiles columns: x0, x1, ..., x19, L
        temp_dict = {"t": tsim}
        for j in range(20):
            temp_dict["x{}".format(j)] = profiles[:, j]
        temp_dict["L"] = profiles[:, 20]

        deduplicate_df = pd.DataFrame(temp_dict)
        deduplicate_df = deduplicate_df.round(4)
        deduplicate_df.drop_duplicates(ignore_index=True, inplace=True)

        # Make dataframe from the simulator results
        t = deduplicate_df["t"]
        x = []
        for j in range(20):
            x.append(deduplicate_df["x{}".format(j)])

        # Note: at this point, x is a list of 20 pandas series, each series has 11 rows
        # Check duplicates were removed correctly
        assert len(t) == 11

        # Make dataframe from the final simulator results
        t = deduplicate_df["t"]
        x = []
        for i in range(20):
            x.append(deduplicate_df["x{}".format(i)])
        L = deduplicate_df["L"]
        u0 = u0_nn
        u1 = u1_nn
        inst_cost = []
        ctg = []

        for time in range(len(t)):
            # Instantaneous cost is L[t1] - L[t0]
            if time == 10:
                inst_cost.append(0)
            else:
                inst_cost.append(L[time + 1] - L[time])

        # Calculate cost to go
        for time in range(len(t)):
            ctg.append(inst_cost[time])
        # Sum backwards from tf
        for time in reversed(range(len(t) - 1)):
            ctg[time] += ctg[time + 1]

        # Calculate path violations
        path = [x[5][int(time * 10)] - 313 for time in t]
        path_violation = []
        for p in path:
            if max(path) > 0:
                path_violation.append(max(path))
            else:
                path_violation.append(p)

        temp_dict = {"t": t}
        for i in range(20):
            temp_dict["x{}".format(i)] = x[i]
        temp_dict["u0"] = u0
        temp_dict["u1"] = u1
        temp_dict["L"] = L
        temp_dict["inst_cost"] = inst_cost
        temp_dict["ctg"] = ctg
        temp_dict["path_diff"] = path_violation

        sindy_sim_results_df = pd.DataFrame(temp_dict)
        sindy_sim_results_df_dropped_tf = sindy_sim_results_df.drop(index=10)

        return sindy_sim_results_df, sindy_sim_results_df_dropped_tf

    def plot(self, dataframe, num_rounds=0, num_run_in_round=0):
        t = dataframe["t"]
        ctg = dataframe["ctg"]

        # Plot x[5] and x[13], the elements whose temperatures we are trying to control
        x5 = dataframe["x5"]
        x13 = dataframe["x13"]
        u0 = dataframe["u0"]
        u1 = dataframe["u1"]

        if "path_diff" in dataframe.columns:
            cst = dataframe["path_diff"]
            if cst.max() <= 0:
                cst_status = "Pass"
            else:
                cst_status = "Fail"
        else:
            cst_status = "None"

        # Check that the cost to go is equal to the Lagrangian cost integral
        assert np.isclose(ctg.iloc[0], dataframe["L"].iloc[-1], atol=0.01)
        total_cost = round(ctg.iloc[0], 3)

        fig, axs = plt.subplots(3, constrained_layout=True)
        fig.set_size_inches(5, 10)

        axs[0].plot(t, x5, label="$x_5$")
        # axs[0].plot(t, np.full(shape=(t.size,), fill_value=313), label="Constraint for $x_5$")
        axs[0].plot(t, np.full(shape=(t.size,), fill_value=303), "--", label="Setpoint for $x_5$")
        axs[0].legend()

        axs[1].plot(t, x13, label="$x_{13}$")
        axs[1].plot(t, np.full(shape=(t.size,), fill_value=333), "--", label="Setpoint for $x_{13}$")
        axs[1].legend()

        axs[2].step(t, u0, label="$u_0$")
        axs[2].step(t, u1, label="$u_1$")
        axs[2].legend()

        # fig.suptitle("Control policy and system state after {} rounds of training \n "
        #              "Run {}: Cost = {}, Constraint = {}"
        #              .format(num_rounds, num_run_in_round, total_cost, cst_status))
        # matplotlib.rc('text', usetex=True)
        fig.suptitle(
            "Sindy MPC Controller: Actual cost achieved = {}\nUsing controls calculated by Sindy MPC on FOM system".format(
                total_cost))
        plt.xlabel("Time")

        # Save plot with autogenerated filename
        svg_filename = results_folder + "svgs/" + "Round {} Run {} Cost {} Constraint {}" \
            .format(num_rounds, num_run_in_round, total_cost, cst_status) + ".svg"
        # plt.savefig(fname=svg_filename, format="svg")
        # plt.savefig(fname="MPC.svg", format="svg")

        plt.show()
        # plt.close()
        return


def generate_trajectories(save_csv=False):
    df_cols = ["t"]
    for i in range(20):
        df_cols.append("x{}".format(i))
    df_cols.extend(["u0", "u1", "L", "inst_cost", "ctg", "path_diff"])
    # 180 trajectories which obeyed the path constraint
    obey_path_df = pd.DataFrame(columns=df_cols)
    # 60 trajectories which violated the path constraint
    violate_path_df = pd.DataFrame(columns=df_cols)
    simple_60_trajectories_df = pd.DataFrame(columns=df_cols)

    num_samples = 0
    num_good = 0
    num_bad = 0

    while num_samples < 240:

        while num_good < 3:
            heatEq_sys = HeatEqSimulator()
            _, trajectory = heatEq_sys.simulate_system_rng_controls()
            if trajectory["path_diff"].max() <= 0:
                simple_60_trajectories_df = pd.concat([simple_60_trajectories_df, trajectory])
                obey_path_df = pd.concat([obey_path_df, trajectory])
                num_good += 1
                num_samples += 1

        while num_bad < 1:
            heatEq_sys = HeatEqSimulator()
            _, trajectory = heatEq_sys.simulate_system_rng_controls()
            if trajectory["path_diff"].max() > 0:
                simple_60_trajectories_df = pd.concat([simple_60_trajectories_df, trajectory])
                violate_path_df = pd.concat([violate_path_df, trajectory])
                num_bad += 1
                num_samples += 1

        # Reset
        num_good = 0
        num_bad = 0

        print("Samples: ", num_samples)

    if save_csv:
        simple_60_trajectories_df.to_csv("heatEq_240_trajectories_df.csv")
        obey_path_df.to_csv("heatEq_obey_path_df.csv")
        violate_path_df.to_csv("heatEq_violate_path_df.csv")


def load_pickle(filename):
    with open(filename, "rb") as model:
        pickled_nn_model = pickle.load(model)
    print("Pickle loaded: " + filename)
    return pickled_nn_model


def replay(trajectory_df_filename, buffer_capacity=360):
    # Use this to keep track where to push out old data
    forgotten_trajectories_count = 0
    pickle_filename = "heatEq_nn_controller_5dim.pickle"
    og_trajectory_df_filename = trajectory_df_filename

    best_cost_after_n_rounds = {}

    for rp_round in range(90):
        trajectory_df = pd.read_csv(results_folder + trajectory_df_filename, sep=",")
        nn_model = load_pickle(pickle_filename)
        run_trajectories = []

        best_cost_in_round = np.inf

        for run in range(6):
            simple_sys = HeatEqSimulator()
            df_1s, df_point9s = simple_sys.simulate_system_nn_controls(nn_model)

            # Store the best result of this run if it passes constraints
            run_cost = df_1s["ctg"][0]
            run_constraint = df_1s["path_diff"].max()
            if run_cost < best_cost_in_round and run_constraint <= 0:
                best_cost_in_round = run_cost

            simple_sys.plot(df_1s, num_rounds=rp_round + 1, num_run_in_round=run + 1)
            run_trajectories.append(df_point9s)

        # Decide whether to push out old memories
        if trajectory_df.shape[0] >= buffer_capacity * 10:
            # Get replace the 6 oldest trajectories with new data (60 rows at a time)
            forgotten_trajectories_count = forgotten_trajectories_count % buffer_capacity
            row_slice_start = forgotten_trajectories_count * 10
            row_slice_end = row_slice_start + 60

            df_temp_concat = pd.DataFrame(columns=trajectory_df.columns.tolist())
            for df_temp in run_trajectories:
                df_temp_concat = pd.concat([df_temp_concat, df_temp])

            trajectory_df.iloc[row_slice_start:row_slice_end] = df_temp_concat.iloc[0:60]
            print(trajectory_df)
            forgotten_trajectories_count += 6

        else:
            for df_temp in run_trajectories:
                trajectory_df = pd.concat([trajectory_df, df_temp])

        trajectory_df_filename = "R{} ".format(rp_round + 1) + og_trajectory_df_filename
        trajectory_df.to_csv(results_folder + trajectory_df_filename)
        pickle_filename = train_and_pickle(rp_round, results_folder + trajectory_df_filename)

        # If best cost in round is better than current running best cost, add it to dictionary
        if len(best_cost_after_n_rounds) == 0:
            best_cost_after_n_rounds[rp_round] = best_cost_in_round
        else:
            best_key = min(best_cost_after_n_rounds, key=best_cost_after_n_rounds.get)
            if best_cost_in_round < best_cost_after_n_rounds[best_key]:
                best_cost_after_n_rounds[rp_round] = best_cost_in_round
            else:
                best_cost_after_n_rounds[rp_round] = best_cost_after_n_rounds[best_key]

        # Plot best cost against rounds
        # Unindent it to plot once, plotting every round and savings just in case anything fails
        plt.plot(*zip(*sorted(best_cost_after_n_rounds.items())))
        plt.title("Best cost obtained after each round")
        plt.xlabel("Number of rounds")
        plt.ylabel("Best cost obtained")
        plot_filename = results_folder + "best_cost_plot.svg"
        plt.savefig(fname=plot_filename, format="svg")
        # plt.show()
        plt.close()

        # Save the best cost against rounds as a csv
        best_cost_csv_filename = results_folder + "best_cost_plot.csv"
        with open(best_cost_csv_filename, "w") as csv_file:
            writer = csv.writer(csv_file)
            for k, v in best_cost_after_n_rounds.items():
                writer.writerow([k, v])

    return


if __name__ == "__main__":
    # generate_trajectories(save_csv=True)

    main_simple_sys = HeatEqSimulator()
    # main_nn_model = load_pickle("simple_nn_controller.pickle")
    main_res, _ = main_simple_sys.simulate_system_nn_controls()
    main_simple_sys.plot(main_res)
    # print(main_res)

    # replay("heatEq_240_trajectories_df.csv")

    # heatEq_system = HeatEqSimulator()
    # main_res, _ = heatEq_system.simulate_system_sindy_controls()
    # heatEq_system.plot(main_res)
    # pd.set_option('display.max_columns', None)
    # print(main_res)
    # main_res.to_csv("heatEq_mpc_trajectory.csv")

    # heatEq_system = HeatEqSimulator()
    # heatEq_system.mpc_control()
    # main_res, _ = heatEq_system.parse_mpc_results()
    # pd.set_option('display.max_columns', None)
    # print(main_res)
    # heatEq_system.plot(main_res)
