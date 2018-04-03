import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import pandas as pd 
#handeling data set
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
X_image = df.iloc[:,1:].values
Y = df.iloc[:,0:1].values
Y = np.reshape(Y,[42000])
# Y = Y.tolist()
# n_values = np.max(Y) + 1
# Y_image = np.eye(n_values)[Y]
X_image = np.reshape(X_image,[42000,1,28,28])
X_image = torch.FloatTensor(X_image)
Y_image = torch.LongTensor(Y)
#defining neural network
X_train = X_image[:37800]
y_train = Y_image[:37800]
X_test = X_image[37800:]
y_test = Y_image[37800:]
#defining neural network
print("ass")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(1,32,3, padding = 1)
        self.conv_2 = nn.Conv2d(32, 32,3, padding = 1 )
        self.conv_3 = nn.Conv2d(32, 64, 3,padding = 1)
        self.conv_4 = nn.Conv2d(64,64, 3 , padding = 1)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7*7*64,128)
        self.fc2 = nn.Linear(128,10)
    def Feedforward(self, x):
        x = F.relu(self.conv_1(x))
        #x = F.relu(self.conv_2(x))
        x = F.max_pool2d(F.relu(self.conv_2(x)),2, stride = 2)
        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(F.relu(self.conv_4(x)),2,stride = 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return x
model = Net()
model.load_state_dict(torch.load('mnist_model6'))
total = 0
for i in range(32):
    correct = 0
    inputs, labels = Variable(X_test[i*128:i*128+128]), Variable(y_test[i*128:i*128+128])
    outputs = model.Feedforward(inputs)
    pred = torch.max(outputs,1)
    correct += (pred[1]==labels).sum()
    print("batch acc out of 128 ",correct)
    total += int(correct)
    #print("total ", total)
print("acc out of 4200 ", total)

