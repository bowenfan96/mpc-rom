# import sklearn.preprocessing
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchinfo import summary

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

# Neural unet structure:
# Xnn: Full model - Intermediate - Reduced
# Unn: Reduced - Intermediate - Full

matrices_folder = "matrices/random_slicot/"
results_folder = "results_csv/random_slicot/"
plots_folder = "results_plots/random_slicot/"


class Encoder(nn.Module):
    def __init__(self, model_dim, reduced_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(model_dim, (model_dim + reduced_dim) // 2)
        self.layer2 = nn.Linear((model_dim + reduced_dim) // 2, (model_dim + reduced_dim) // 4)
        self.layer3 = nn.Linear((model_dim + reduced_dim) // 4, reduced_dim)

        nn.init.kaiming_uniform_(self.layer1.weight)
        nn.init.kaiming_uniform_(self.layer2.weight)
        nn.init.kaiming_uniform_(self.layer3.weight)

    def forward(self, x_in):
        # x_in = torch.flatten(data, start_dim=1)
        # Input to intermediate layer activation
        xe1 = F.leaky_relu(self.layer1(x_in))
        xe2 = torch.sigmoid(self.layer2(xe1))
        x_rom = (self.layer3(xe2))
        return x_rom


class Decoder(nn.Module):
    def __init__(self, model_dim, reduced_dim):
        super(Decoder, self).__init__()
        self.layer4 = nn.Linear(reduced_dim, (model_dim + reduced_dim) // 4)
        self.layer5 = nn.Linear((model_dim + reduced_dim) // 4, (model_dim + reduced_dim) // 2)
        self.layer6 = nn.Linear((model_dim + reduced_dim) // 2, model_dim)

        nn.init.kaiming_uniform_(self.layer4.weight)
        nn.init.kaiming_uniform_(self.layer5.weight)
        nn.init.kaiming_uniform_(self.layer6.weight)

    def forward(self, x_rom):
        xd1 = F.leaky_relu(self.layer4(x_rom))
        xd2 = torch.sigmoid(self.layer5(xd1))
        x_dec = (self.layer6(xd2))
        return x_dec


class Wrapper(nn.Module):
    def __init__(self, model_dim, reduced_dim):
        super(Wrapper, self).__init__()
        self.encoder = Encoder(model_dim, reduced_dim)
        self.decoder = Decoder(model_dim, reduced_dim)

    def forward(self, x_in):
        x_rom = self.encoder(x_in)
        x_out = self.decoder(x_rom)
        return x_out


class Autoencoder():
    def __init__(self, data, config=None):
        super(Autoencoder, self).__init__()

        # Hyperparameters
        if config is not None:
            self.nb_epoch = config["num_epochs"]
            self.batch_size = config["batch_size"]
            self.learning_rate = config["learning_rate"]

            # Desired dimension of reduced model
            self.reduced_dim_size = config["x_rom"]
            # We report loss on validation data to RayTune if we are in tuning mode
            self.is_tuning = True

        # If we are not tuning, then set the hyperparameters to the optimal ones we already found
        else:
            self.num_epoch = 1500
            self.batch_size = 50
            self.learning_rate = 0.005
            self.reduced_dim_size = 200
            self.is_tuning = False

        # Initialise parameters
        processed_data = self.process_data(data)

        # Input size is the same as output size
        self.model_dim = processed_data.shape[1]

        print(self.model_dim)
        print(self.reduced_dim_size)

        # Initialise autoencoder neural unet
        self.autoencoder = Wrapper(self.model_dim, self.reduced_dim_size)

        return

    def process_data(self, data):
        # Convert to torch tensors
        # print("initial data")
        # print(data)
        self.x_scaler = preprocessing.MinMaxScaler()
        self.x_scaler.fit(data)
        data = self.x_scaler.transform(data)

        # print("scaled data")
        # print(data)
        #
        # time.sleep(10)
        data = torch.tensor(data, dtype=torch.float)
        # print(data)
        return data

    def fit(self, data):
        # Process data, convert pandas to tensor
        data = self.process_data(data)

        # Set model to training model so gradients are updated
        self.autoencoder.train()

        # Wrap the tensors into a dataset, then load the data
        # data = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)

        # Create optimizer to use update rules
        optimizer = optim.SGD(self.autoencoder.parameters(), lr=self.learning_rate)

        # Specify criterion used
        criterion = nn.MSELoss()

        # Train the neural network
        for epoch in range(self.num_epoch):

            # Full dataset gradient descent
            # output = self.autoencoder(data)
            # loss = criterion(output, data)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # Minibatch gradient descent
            for minibatch_data in data_loader:
                optimizer.zero_grad()
                output = self.autoencoder(minibatch_data)
                # print(output)
                loss = criterion(output, minibatch_data)
                loss.backward()
                optimizer.step()

            # Test entire dataset at this epoch
            with torch.no_grad():
                # print("Data in")
                # print(data)
                # print("Data out")
                output = self.autoencoder(data)
                # print(output)
                loss = criterion(output, data)

            # Print loss
            print('The loss of epoch ' + str(epoch) + ' is ' + str(loss.item()))

        return self

    def predict(self, x_in):
        # Scale data
        x_in = self.x_scaler.transform(np.array(x_in).reshape(1, 200))
        x_in = torch.tensor(x_in, dtype=torch.float)
        with torch.no_grad():
            x_out = self.autoencoder(x_in)
        x_out = self.x_scaler.inverse_transform(x_out)
        print(x_out)
        return x_out


def autoencoder_train():
    data = pd.read_csv(results_folder + "all.csv", sep=',')

    autoencoder = Autoencoder(data)

    autoencoder.fit(data)

    # Print a model.summary to show hidden layer information
    summary(autoencoder.autoencoder.to("cpu"), verbose=2)

    with open('autoencoder.pickle', 'wb') as model:
        pickle.dump(autoencoder, model)
    print("\nSaved model to autoencoder.pickle\n")


def autoencoder_test():
    with open('autoencoder.pickle', 'rb') as model:
        autoencoder = pickle.load(model)

    # data = [-14.83165629,-161.1006275,17.97470927,47.14182177,-162.2690421,78.29780601,-193.0152237,-130.4433429,-79.83324802,-177.6648203,-59.20701185,-316.8436072,581.7091576,-364.4575812,-55.50943308,259.0067225,-18.79825353,137.7720616,-113.5622711,47.47252004,361.5519402,27.11267485,-181.4168155,389.3637661,-227.269274,210.4659172,153.3932786,135.9427983,-192.4188522,37.62288008,-42.644953,-303.4752865,543.930331,84.5999829,278.4498494,116.2224334,459.8991341,80.18154251,-213.8999713,-179.6900801,-133.5569977,41.06426457,176.7759106,83.29979494,229.8118634,401.3875316,-109.3206292,-229.3748391,-820.2839729,398.1342629,123.8113763,54.34533879,329.8212805,-570.8532086,-175.9389129,-192.4876244,6.787838322,-16.22951519,-152.1703322,501.5753162,131.385075,-52.06404119,57.51980978,-346.6791255,13.38111854,-51.42959382,270.2798148,-82.62607923,193.2351923,360.5169641,395.4452415,-47.96863634,740.2787067,471.7195239,37.29533813,-71.26449988,372.7970232,-24.86907709,227.8411487,-77.25209014,-204.7795776,-187.2873872,332.0304124,27.91654651,357.3568502,-148.4332419,353.8629867,23.14209096,7.202166796,203.6242851,-210.1658523,51.90868387,49.67711498,-100.5203153,-138.2994049,188.9026341,0.413971399,3.213608079,124.9050301,107.5878206,34.54029538,-15.51850652,-204.3204602,-38.82523513,42.83683967,-35.16427024,53.69048921,-110.2029605,-189.0583913,80.24854789,-49.36930532,-6.070158996,-81.09506131,11.40523904,-11.84949353,-15.36728196,-2.14419194,56.79011355,136.1980812,175.4618878,186.8086246,-238.4190614,-260.8073908,79.30515515,244.8677373,-79.4025007,14.85806438,-232.5752048,-390.5442475,129.1900585,-2.064770754,180.9149099,-436.0893772,-31.16618229,-69.81843743,125.7258897,-209.0663268,-12.87401065,26.04354631,-106.5268154,-110.2877255,-219.1026993,-114.6741264,-45.57122979,9.53584334,-76.97356896,144.0334187,97.93923142,98.05194751,-196.5300541,61.51559435,461.5577533,-214.391747,-6.058039881,391.8946142,69.45064012,268.388253,-118.364413,171.8923337,35.82528264,130.1407027,-419.0155452,-207.5311786,11.22711036,218.3150946,-237.2247592,22.82488816,171.3641412,156.0240522,247.8732219,-113.4816532,23.99256827,-6.477729128,216.5553373,-227.1859681,140.1464121,-108.1599508,-128.1151378,85.04492621,118.9456466,74.69584921,-163.0141505,29.76430145,74.49849933,-17.17116426,373.2227229,119.3814386,116.0971489,-172.1071067,-159.4056265,186.991424,282.6237733,78.22459926,-70.7741115,78.11667007,66.40763088,92.24720305,-253.8337271,96.5695002,163.1712271]
    # data = [0,-3,2,5,-4,-1,-2,-1,-6,-6,7,1,5,3,-5,0,7,-5,-1,7,7,9,-3,-1,0,7,-8,0,-1,0,-3,2,2,0,-3,-8,-10,-4,6,-7,-1,-4,4,-6,-6,4,4,-8,1,9,-9,-10,3,2,5,0,1,1,-10,-1,9,5,-9,1,0,9,2,5,0,-4,-7,-1,-9,2,3,5,5,3,-2,-1,-9,8,3,6,8,6,-8,-9,-10,9,6,2,9,3,-10,6,-8,8,1,8,2,1,-5,8,2,3,-3,4,-10,-3,0,5,7,2,-3,-1,6,-10,-9,-7,-4,0,5,0,-3,-5,7,-7,7,3,-9,9,-6,-3,-3,-10,1,-3,0,-4,-5,4,9,3,-9,-1,-7,-6,-1,8,-5,6,-10,-7,1,-8,-2,5,-2,-7,-6,-9,-1,-7,-3,-1,4,9,-8,-6,9,-8,-3,-10,2,8,-2,-5,-6,1,6,7,-9,3,-6,3,3,-10,-10,6,5,-1,8,4,1,-8,-2,-4,1,7]
    data = [-0.000380781,-0.016082756,0.001606849,0.006598728,-0.015424072,0.008972052,-0.00688947,-0.012306712,-0.010226534,-0.016428648,-0.00325512,-0.035478145,0.063991183,-0.034026988,-0.005049429,0.023786426,0.000238768,0.028544006,-0.011090252,0.005889204,0.039271504,-0.002113556,-0.015729859,0.038619359,-0.024211898,0.022305543,0.017172088,0.008593284,-0.02251253,-0.000839726,-0.008091669,-0.034634303,0.042694103,0.012970235,0.029621393,0.011086759,0.051551876,0.002188725,-0.01979643,-0.016204995,-0.015202358,0.00137466,0.020509109,0.010753877,0.016108803,0.05002563,-0.009338412,-0.033431232,-0.082699851,0.053059624,0.007670459,0.007704934,0.028714796,-0.073376113,-0.020224299,-0.025124238,-0.001952411,0.003230624,-0.013905312,0.040416053,0.013574863,-0.00461624,0.010761351,-0.040069498,0.008534925,-0.007510644,0.025512425,-0.002787428,0.022682888,0.035873348,0.03839632,-0.004496113,0.0651386,0.055999939,0.005847371,-0.00212087,0.025128899,0.00188984,0.030407225,-0.008077534,-0.026044724,-0.014568357,0.041512798,-0.002952083,0.033301801,-0.016618226,0.036868106,-5.71E-05,-0.001447632,0.024715878,-0.018098159,0.01208182,0.003132128,-0.010987474,-0.009285843,0.015626083,-0.003320361,0.001644939,0.016623299,0.017834251,0.007481852,-0.00194825,-0.021915183,-0.008778856,0.006712105,-0.006888142,-0.002310878,-0.015131904,-0.021848181,0.006910442,-0.007003452,-0.005787779,-0.007400702,-0.004479156,0.007988632,0.002993569,-0.00183758,0.010801819,0.017888783,0.030347755,0.023279789,-0.027644773,-0.031344575,0.014576077,0.031821682,-0.013391926,-0.001146905,-0.019203808,-0.053393758,0.019512208,0.003707271,0.025810156,-0.053952864,-0.001341582,-0.006252042,0.012888136,-0.025872012,0.001628538,0.008782096,-0.006622979,-0.013007476,-0.023688954,-0.014628948,-0.006023674,-0.005127105,-0.005794106,0.013602289,0.00166714,0.012579874,-0.021605221,0.006311895,0.050795446,-0.023604764,-0.001679105,0.037648052,0.008194962,0.026445564,-0.02419427,0.017975216,0.009140339,0.018225189,-0.044045088,-0.028472688,-0.001899544,0.023439513,-0.027825759,-0.001189318,0.019006555,0.022310369,0.032868984,-0.027424082,-0.000152061,0.009873914,0.028156255,-0.024755093,0.015225695,-0.010009479,-0.009821093,0.000947325,0.007779837,0.012167215,-0.012317023,0.012375894,-0.000505999,-0.000865928,0.046974707,0.005585724,0.012321519,-0.01742988,-0.016204681,0.023500507,0.028494946,0.004644275,-0.013335618,0.004997101,0.004843276,0.012172503,-0.034691985,0.017582278,0.018328701]
    autoencoder.predict(data)


if __name__ == "__main__":
    autoencoder_train()
    autoencoder_test()

    criterion = nn.MSELoss()
    x_in = [[0.9474, 0.3684, 0.4211, 0.9474, 1.0000, 0.6842],
            [0.5263, 0.5255, 0.5264, 0.5245, 0.5272, 0.5273],
        [0.5257, 0.5197, 0.5270, 0.5160, 0.5303, 0.5330],
        [0.5248, 0.5095, 0.5282, 0.4997, 0.5365, 0.5434],
        [0.5234, 0.4943, 0.5299, 0.4758, 0.5456, 0.5588],
        [0.5215, 0.4742, 0.5321, 0.4441, 0.5576, 0.5791]]

    x_out = [[0.5381, 0.5326, 0.5353, 0.5150, 0.5435, 0.5238],
        [0.5188, 0.5105, 0.5268, 0.4965, 0.5362, 0.5408],
        [0.5182, 0.5094, 0.5267, 0.4969, 0.5369, 0.5409],
        [0.5171, 0.5075, 0.5265, 0.4977, 0.5384, 0.5410],
        [0.5154, 0.5047, 0.5262, 0.4990, 0.5405, 0.5411],
        [0.5128, 0.5013, 0.5256, 0.5002, 0.5434, 0.5402]]

    x_in = torch.tensor(x_in)
    x_out = torch.tensor(x_out)

    print(criterion(x_in, x_out))