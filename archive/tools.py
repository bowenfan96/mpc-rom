import control
import numpy
import numpy as np

import pandas as pd
import glob

# from mor_nn import *
# from mor_nn_simple import *
from simple_ctg_nn import *

import torch.utils.data

# from deep_nn import *
import pickle
import matplotlib.pyplot as plt

# x_dot = Ax + Bu
# y = Cx + Du

numpy.set_printoptions(suppress=True, precision=3, linewidth=250)

matrices_folder = "matrices/random_slicot/"
results_folder = "results_csv/random_slicot/"
plots_folder = "results_plots/random_slicot/"


def generate_model():
    controllable = False
    observable = False

    while not controllable or not observable:
        A = numpy.random.randint(-10, 10, size=(5, 5))
        B = numpy.random.randint(-20, 20, size=(5, 2))
        C = numpy.random.randint(-10, 10, size=(1, 5))
        D = numpy.random.randint(-20, 20, size=(2, 1))

        ctrb = control.ctrb(A, B)
        obsv = control.obsv(A, C)

        length_A = max(A.shape)
        ctrl_rank = numpy.linalg.matrix_rank(ctrb)
        obsv_rank = numpy.linalg.matrix_rank(obsv)

        if (length_A - ctrl_rank) == 0 and (length_A - obsv_rank) == 0:
            controllable = True
            observable = True
            numpy.savetxt("A.csv", A, delimiter=",")
            numpy.savetxt("B.csv", B, delimiter=",")
            numpy.savetxt("C.csv", A, delimiter=",")
            numpy.savetxt("D.csv", B, delimiter=",")

        print(ctrl_rank, obsv_rank)


def concat_csv():
    path = results_folder   # use your path
    all_files = glob.glob(path + "*.csv")

    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)

    all_results = pd.concat(df_list, axis=0, ignore_index=True)
    all_results.to_csv(results_folder + "all_simple.csv")


def reshape_csv():
    file = "simple_system_unsorted_data/simple_proper_rng_controls_init_fix.csv"
    df = pd.read_csv(file, index_col=None, header=0)
    df.drop(index=df.iloc[::11, :].index.tolist(), inplace=True)
    df.to_csv("simple_proper_rng_controls_init_fix_clean.csv")


def plot_u_vs_ctg():
    # Load pickle neural unet
    with open('simple_system_unsorted_data/simple_proper_wctg.pickle', 'rb') as model:
        mor_nn = pickle.load(model)
    u_full = np.linspace(start=-20, stop=20, num=41)

    # x_full = np.random.randint(low=800, high=1000, size=200).reshape(1, -1)
    # x_full = [-14.8316562931729,-161.10062750446366,17.9747092734893,47.14182177395436,-162.26904208293084,78.29780601109103,-193.01522368267962,-130.44334293429574,-79.8332480158878,-177.66482029137472,-59.207011846674234,-316.8436072236877,581.7091576198566,-364.45758119944713,-55.50943307795587,259.0067225247527,-18.79825352720716,137.77206160695346,-113.56227108238087,47.47252003521275,361.55194021034134,27.112674845798285,-181.41681553691916,389.36376609504833,-227.2692740095383,210.4659172401764,153.39327863037911,135.94279829105668,-192.41885224264735,37.62288007967707,-42.6449529959054,-303.4752864868022,543.9303310242612,84.59998290281035,278.4498493872987,116.22243335272843,459.8991341304475,80.18154251148033,-213.89997132139004,-179.69008008618925,-133.55699765678224,41.06426456757407,176.7759105806744,83.29979493560216,229.8118633837092,401.38753164162193,-109.32062924294236,-229.37483910418484,-820.2839728671887,398.13426287752293,123.81137629545518,54.345338787720465,329.82128049726174,-570.8532085793858,-175.93891292036307,-192.48762436991234,6.787838322165642,-16.229515188552167,-152.17033223165322,501.5753161830803,131.38507499323987,-52.064041185590476,57.519809783114034,-346.679125529568,13.381118542355097,-51.42959381947433,270.2798148368617,-82.62607923313372,193.23519230781073,360.5169640627437,395.4452415206668,-47.96863634440144,740.2787066960268,471.71952387236814,37.29533813256822,-71.26449988204477,372.79702321039946,-24.8690770928971,227.84114865973592,-77.25209013624107,-204.77957764002818,-187.28738715776774,332.03041240288877,27.916546509060687,357.35685018094006,-148.43324190661173,353.8629866822482,23.142090961790796,7.20216679570412,203.62428513702574,-210.1658522508241,51.908683872939335,49.677114983252494,-100.52031532031677,-138.29940486285798,188.90263407953657,0.4139713988850332,3.213608078940879,124.90503012690827,107.58782061322295,34.54029537528566,-15.518506520552608,-204.32046020075904,-38.825235131041794,42.8368396662026,-35.164270244477684,53.69048920764253,-110.20296048639483,-189.0583913215663,80.24854789443289,-49.369305319978466,-6.070158995698681,-81.09506131046149,11.405239036661477,-11.849493534725754,-15.367281955192956,-2.1441919397448252,56.790113548093,136.1980811539681,175.46188776611933,186.80862457492552,-238.4190614413196,-260.80739078418765,79.30515514821042,244.86773734923335,-79.40250070177072,14.858064381993813,-232.57520479573893,-390.54424753113784,129.19005846103494,-2.064770753911862,180.9149098972705,-436.08937721324673,-31.16618229264169,-69.81843743124885,125.72588967940118,-209.0663267554714,-12.874010646983704,26.04354630791947,-106.52681544288795,-110.28772550874314,-219.10269931926743,-114.67412643382463,-45.5712297935143,9.53584333955126,-76.97356895673065,144.0334186734306,97.93923142048622,98.0519475096514,-196.53005407915967,61.51559434884747,461.5577532874313,-214.39174699159759,-6.058039880851658,391.8946142149573,69.45064011770333,268.3882530359976,-118.36441297668051,171.89233365804344,35.82528263800186,130.1407027246673,-419.0155451885467,-207.53117856501683,11.227110357345406,218.315094646534,-237.22475917873518,22.824888163371714,171.36414118636404,156.0240522051755,247.87322185044417,-113.48165319461498,23.992568270107906,-6.477729127503553,216.55533730497748,-227.18596810309126,140.1464120754937,-108.15995075291015,-128.1151377761713,85.04492620975591,118.94564658216758,74.69584921397707,-163.01415049755758,29.764301445036455,74.49849932945456,-17.171164256872768,373.2227228793056,119.38143858755136,116.09714890386253,-172.1071066807875,-159.40562651945004,186.9914239752579,282.62377328335026,78.22459925542226,-70.77411149888584,78.11667006639688,66.40763088419216,92.24720305106577,-253.83372707720005,96.5695002023225,163.1712271339642]
    # x_full = np.array(x_full).reshape(1, -1)
    x_full = [0, 1]
    x_full = np.array(x_full).reshape(1, -1)
    u_rom_plot = []
    ctg = []
    for u in u_full:
        print(u)
        # u_rom = mor_nn.encode_u(u)
        # u_rom_plot.append(u_rom.flatten())
        ctg.append(mor_nn.predict_ctg(x_full, u))

    plt.plot(u_full, ctg)
    plt.title("Cost to go against u, fixing x1={} and x2={}".format(x_full[0,0], x_full[0,1]))
    # plt.plot(u_full, u_rom_plot)
    plt.show()


def check_ctg():
    # Load pickle neural unet
    with open('mor_nn_simple.pickle', 'rb') as model:
        mor_nn = pickle.load(model)

    x_full = [4, 1]
    x_full = np.array(x_full).reshape(1, -1)
    u = [2]

    print("Predicted ctg")
    print(mor_nn.predict(x_full, u))


def predict_state_and_controls():
    with open('deep_nn_simple.pickle', 'rb') as model:
        deep_nn_ = pickle.load(model)

    x_full = [4, 1]
    # x_full = np.array(x_full).reshape(1, -1)
    u = [2]

    x_full = torch.FloatTensor(x_full)
    u = torch.FloatTensor(u)

    xu_i = torch.hstack((x_full, u))

    xu_f_pred = deep_nn_.predict(xu_i)

    print("Predicted")
    print(xu_f_pred)


def append_ctg():
    data = pd.read_csv("simple_system_unsorted_data/simple_proper_rng_controls.csv", sep=',')
    # split_points = data.iloc[10::11, :]
    # print(split_points)

    split_points = np.arange(start=11, stop=8680, step=11)
    split_df = np.split(data, split_points)

    # print(split_df)

    def cost(x1, x2, u):
        # setpoint_cost = (y_row - 100) ** 2
        # setpoint_cost = sum((50 - xi)**2 for xi in x_row)
        # controller_cost = sum(ui**2 for ui in u_row)
        # controller_cost = u_row ** 2
        # weighted_cost = 0.8 * setpoint_cost + 0.2 * controller_cost
        # return weighted_cost
        # return setpoint_cost
        return x1**2 + x2**2 + 5E-3*u**2

    data_w_ctg = []

    for df in split_df:
        cost_to_go = []
        df = np.array(df)
        for t in range(df.shape[0]):
            cost_to_go.append(cost(df[t][2], df[t][3], df[t][4]))
        for t in reversed(range(df.shape[0] - 1)):
            cost_to_go[t] += cost_to_go[t + 1]
        cost_to_go = np.array(cost_to_go).reshape(-1, 1)
        w_ctg = np.hstack((df, cost_to_go))
        data_w_ctg.append(w_ctg)

    print(data_w_ctg)

    df_export = pd.DataFrame(np.vstack(data_w_ctg))

    df_export.drop(index=df_export.iloc[0::11, :].index.tolist(), inplace=True)
    print(df_export)

    df_export.to_csv("simple_proper_rng_controls_wctg.csv")


if __name__ == "__main__":
    # predict_state_and_controls()
    # concat_csv()
    # reshape_csv()
    # check_ctg()
    plot_u_vs_ctg()

    # append_ctg()