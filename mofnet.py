
from torch._C import device
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.utils.data
import math
import pandas as pd
import numpy as np
import scipy as sp
import random
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.metrics import recall_score,precision_score,roc_auc_score,roc_curve
import matplotlib
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoFNetLayer(nn.Module):
    def __init__(self, in_dims, out_dims, bias=True):
        super(MoFNetLayer, self).__init__()
        self.in_dims = in_dims
        self.in_dims = out_dims
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        return input.matmul(self.weight.t() * adj) + self.bias

    def extra_repr(self):
        return 'in_dims={}, out_dims={}, bias={}'.format(
            self.in_dims, self.out_dims, self.bias is not None
        )

def preprocess(x, y):
    return x.float().to(device), y.int().reshape(-1, 1).to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

class Net(nn.Module):
    def __init__(self, adj_1, adj_2, D_in, T1, H1, H2, H3, D_out):
        super(Net, self).__init__()
        self.adj_1 = adj_1  # only gene expression with snp
        self.adj_2 = adj_2  # only gene expression with protein expression 
        self.MoFNet1 = MoFNetLayer(D_in, T1)
        self.MoFNet2 = MoFNetLayer(T1+H1, H1)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        t1 = self.MoFNet1(x[:, 186:1751], self.adj_1).relu()
        x_2 = torch.cat((x[:, 0:186], t1),1)
        h1 = self.MoFNet2(x_2, self.adj_2).relu()
        h1 = self.dropout1(h1)
        h2 = self.linear2(h1).relu()
        h2 = self.dropout1(h2)
        h3 = self.linear3(h2).relu()
        y_pred = self.linear4(h3).sigmoid()
        return y_pred

def loss_batch(model, loss_fn, xb, yb, opt=None):
    yhat = model(xb)
    loss = loss_fn(yhat, yb.float())
    for param in model.MoFNet1.parameters():
            loss += L1REG * torch.sum(torch.abs(param))
    

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    yhat_class = np.where(yhat.detach().cpu().numpy()<0.5, 0, 1)
    accuracy = accuracy_score(yb.detach().cpu().numpy(), yhat_class)

    return loss.item(), accuracy

def fit(epochs, model, loss_fn, opt, train_dl, val_dl):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    for epoch in range(epochs):
        model.train()
        losses, accuracies = zip(
            *[loss_batch(model, loss_fn, xb, yb, opt) for xb, yb in train_dl]
        )
        train_loss.append(np.mean(losses))
        train_accuracy.append(np.mean(accuracies))

        model.eval()
        with torch.no_grad():
            losses, accuracies = zip(
                *[loss_batch(model, loss_fn, xb, yb) for xb, yb in val_dl]
            )
        val_loss.append(np.mean(losses))
        val_accuracy.append(np.mean(accuracies))
        
        if (epoch % 10 == 0):
            print("epoch %s" %epoch, np.mean(losses),np.mean(train_accuracy), \
                  np.mean(accuracies))
    
    return train_loss, train_accuracy, val_loss, val_accuracy

n_seed = 66
np.random.seed(n_seed)
torch.manual_seed(n_seed)
random.seed(n_seed)

data_path = '**' #Set the path

#Loading the data
X_train = pd.read_csv(data_path + '*.csv',header=None)
X_val = pd.read_csv(data_path + '*.csv',header=None)
X_test = pd.read_csv(data_path + '*.csv',header=None)
y_train = pd.read_csv(data_path + '*.csv',header=None)
y_val = pd.read_csv(data_path + '*.csv',header=None)
y_test = pd.read_csv(data_path + '*.csv',header=None)
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

#Loading the adjacency matrix for MoFNet layer 1 and layer 2
adj1 = pd.read_csv(data_path + '*.csv')
adj1 = adj1.set_index('probe')
adj1 = np.array(adj1)

adj2 = pd.read_csv(data_path + '*.csv')
adj2 = adj2.set_index('probe')
adj2 = np.array(adj2)


X_train, y_train, X_val, y_val = map(torch.tensor,(X_train,y_train,X_val,y_val))


train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)


train_dl = DataLoader(dataset=train_ds, batch_size=30) 
val_dl = DataLoader(dataset=val_ds) 

train_dl = WrappedDataLoader(train_dl, preprocess)
val_dl = WrappedDataLoader(val_dl, preprocess)

#Define the H2,H3 size and other params
H2 = 96
H3 = 16
L1REG = 0.005
L2REG = 0.0008
LR = 0.001
epochs = 100
loss = nn.BCELoss()
opt = torch.optim.Adam

D_in, T1, H1, H2, H3, D_out = 743+822, 743, 186, H2, H3, 1
adj_1 = torch.from_numpy(adj1).float().to(device)
adj_2 = torch.from_numpy(adj2).float().to(device)

model = Net(adj_1, adj_2, D_in, T1, H1, H2, H3, D_out).to(device)


weight_decay=L2REG
opt = opt(model.parameters(), lr=LR, weight_decay=L2REG, \
          betas=(0.9, 0.999),amsgrad=True)
train_loss, train_accuracy, val_loss, val_accuracy = fit(epochs, \
          model, loss, opt, train_dl, val_dl)
fig, ax = plt.subplots(2, 1, figsize=(8,4))

ax[0].plot(train_loss)
ax[0].plot(val_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss,H2--{0},H3--{1},'.format(H2,H3))

ax[1].plot(train_accuracy)
ax[1].plot(val_accuracy)
ax[1].legend(labels=['Train','Test'])
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

print('Training process has finished. Saving trained model.')
print('Starting testing')
torch.save(model, "/*.pth")


with torch.no_grad():
  x_tensor_test = torch.from_numpy(X_test).float().to(device)
  model.eval()
  yhat = model(x_tensor_test)
  y_hat_class = np.where(yhat.cpu().numpy()<0.5, 0, 1)
  test_accuracy = accuracy_score(y_test.reshape(-1,1), y_hat_class)
  f1 = f1_score(y_test.reshape(-1,1), y_hat_class)
  recall = recall_score(y_test.reshape(-1,1), y_hat_class)
  precision = precision_score(y_test.reshape(-1,1), y_hat_class)
  fpr, tpr, threshold = roc_curve(y_test.reshape(-1,1), y_hat_class)
  auc_score = roc_auc_score(y_test.reshape(-1,1), y_hat_class)
  tn, fp, fn, tp = confusion_matrix(y_test.reshape(-1,1), y_hat_class).ravel()
  specificity = tn / (tn+fp)

          
  print('Accuracy: %d %%' % (100.0 * test_accuracy))
  print("auc_score:",auc_score)
  print("specificity:",specificity)
  print("Recall:",recall)
  print("Precision:",precision)
  print("F1 score:",f1)
  print('--------------------------------')

!pip install captum

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

#Loading your Model
model = torch.load("*.pth",map_location=torch.device('cpu'))

input_feature_name = list(adj_1.index) # for layer 1 adj1 and for layer 2 adj2
model.adj_1 = model.adj_1.cpu() #change adj_1 -> adj_2 for layer 2
intergrated = IntegratedGradients(model.cpu())
test_input = torch.from_numpy(X_test).type(torch.FloatTensor)
attr, delta = intergrated.attribute(test_input, return_convergence_delta=True)
attr = attr.detach().numpy()

importances = dict(zip(input_feature_name, np.mean(abs(attr), axis=0)))

outFile = 'layer1_importance_of_input_feature.csv'

print('Iuput feature importance {}'.format(outFile))
with open(outFile, 'w') as f:
    for key in importances.keys():
        f.write("%s,%s\n"%(key,importances[key]))

cond = LayerConductance(model, model.MoFNet1) #for layer 2 change it to MoFNet2

cond_vals = cond.attribute(test_input)
cond_vals = cond_vals.detach().numpy()

importances_layer1 = dict(zip(adj1.columns.tolist(), \
            np.mean(abs(cond_vals), axis=0))) #change adj1 -> adj2 for layer 2

outFile = 'layer1_node_importance.csv'

print('Transparent layer node importance {}'.format(outFile))
with open(outFile, 'w') as f:
    for key in importances_layer1.keys():
        f.write("%s,%s\n"%(key,importances_layer1[key]))

#for layer 2 change it to MoFNet2
neuron_cond = NeuronConductance(model, model.MoFNet1) 
outFile = 'layer1_connection_weights.csv'
with open(outFile, 'w') as f:
    print('Connection weights')
    #change adj1 -> adj2 for weight extraction of layer 2 
    for idx in adj1.columns.tolist(): 
        neuron_cond_vals = neuron_cond.attribute(test_input, \
            neuron_selector=adj1.columns.tolist().index(idx)) 
        importances_neuron = dict(zip(input_feature_name, \
            abs(neuron_cond_vals.mean(dim=0).detach().numpy())))
        importances_neuron = {key:val for key, val in \
                              importances_neuron.items() if val != 0}
        for key in importances_neuron.keys():
            f.write("%s,%s,%s\n"%(idx,key,importances_neuron[key]))
