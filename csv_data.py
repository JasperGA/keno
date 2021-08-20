import torch
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import Network
# small change

def train(net, train_loader, val_loader, optimizer):
    total=0
    correct=0
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()    # zero the gradients
        output = net(data)       # apply network
        loss = F.binary_cross_entropy(output,target)
        loss.backward()          # compute gradients
        optimizer.step()         # update weights
        pred = (output >= 0.5).float()
        correct += (pred == target).float().sum()
        total += target.size()[0]
        accuracy = 100*correct/total

    if epoch % 100 == 0:
        print('ep:%5d loss: %6.4f acc: %5.2f' %
             (epoch,loss.item(),accuracy))
        val_acc = 0.0
        correct = 0
        with torch.no_grad():           ######################################### CHECK THIS VALIDATION (seems right now maybe?)
            net.eval()
            for inputs, labels in val_loader:               # MAKE IT OUTPUT ITS PREDICTION AND CHECK IT MANUALLY, SEEMS TOO GOOD TO BE TRUE
                predicts = net(inputs)                      # MUST BE OVERFITTING MAYBE
                pred = (predicts >= 0.5).float()
                #print(pred)
                correct += (pred == labels).float().sum()
                val_acc += correct.item()
        
            net.train()
        
        #print(val_acc)
        print('Validation Acc: {:.6f}'.format(val_acc / (20)))  # MODIFY THIS NUMBER HERE

    return accuracy


numbers = pd.read_csv('data\Results_08072021_200_200.csv', usecols = [i for i in range(1, 21)])

#check = numbers.head(5)
#print(numbers)
#print(numbers.shape[0])
#print(check)

#print(numbers.values[0][0])

#   Bellow processes numbers
#  takes keno numbers from there numbers into a grid of [80 x 1] ones and zeros (1 means that number, 0 means number not there)
# stored in results
results = np.zeros((numbers.shape[0]-1, 80))    # numbers.shape[0]-1 cause don't know result of game after the last game in dataset

row = 0
col = 0
while row < numbers.shape[0]-1:
    while col != 20:
        num = numbers.values[row][col]
        results[row][num - 1] = 1           # need num - 1 cause arrays start at 0 (num=4 would index at 5 instead)
        col += 1
    
    col = 0
    row +=1

########################################################
# Splits data into groups of 4 games
# Training splitting
matrixes = int(numbers.shape[0]/4)
print(matrixes)
print("HERE")
split_4 = np.array_split(results, matrixes)
split_4 = split_4[0:(len(split_4)-1)] 

train_4 = np.zeros([matrixes-1, 4, 80])

arr = 0         
for arr2d in split_4:
    train_4[arr] = arr2d
    arr += 1

print(train_4[0][0])
print(results[0])

###########################################################

# Attemptting to class if 1 will appear in next game (just looking at previous game for now)

#print(results.shape[0])
targets = np.zeros([results.shape[0], 1])     # if 1 appears in next game, model should output 1, if not then output 0
                                                # use results.shape[0] because we need same number of targets as sets

num_row = 0
num_target = 0
while num_row < results.shape[0]-1:
    targets[num_target][0] = results[num_row + 1][0]
    
    num_row = num_row + 1
    num_target = num_target + 1
    
#print(results.shape)
#print(targets.shape)

######################################################
# Target splitting for 4 games at a time
target_4 = np.zeros([len(split_4), 1])
num = 0
while num != len(split_4):
    target_4[num][0] = targets[4*(num+1)-1][0]
    num += 1

print('TART')
print(target_4[1])
print(targets[7])

print(target_4.shape)
print(train_4.shape)


# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--init', type=float,default=0.1, help='initial weight size')
parser.add_argument('--lr', type=float,default=0.01, help='learning rate')
parser.add_argument('--epoch', type=int,default='10000', help='max training epochs')
args = parser.parse_args()


full_input = torch.from_numpy(train_4)
full_target = torch.from_numpy(target_4)

full_input = full_input.to(torch.float32)
full_target = full_target.to(torch.float32)

train_input = full_input[0:30, :]           # CHECK THIS INDEXING ############################################################
val_input = full_input[29:49, :]

train_target = full_target[0:30, :]         
val_target = full_target[29:49, :]

#print('val')
#print(val_target)
#print('val')

train_dataset = torch.utils.data.TensorDataset(train_input,train_target)
train_loader  = torch.utils.data.DataLoader(train_dataset,batch_size=98)

val_dataset = torch.utils.data.TensorDataset(val_input, val_target)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=98)


# choose network architecture
net = Network()
    
if list(net.parameters()):
    # initialize weight values
    for m in list(net.parameters()):
        m.data.normal_(0,args.init)

            # use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(),eps=0.000001,lr=args.lr,
                                 betas=(0.9,0.999),weight_decay=0.0001)

    # training loop
    epoch = 0
    count = 0
    while epoch < args.epoch and count < 2000:
        epoch = epoch+1
        accuracy = train(net, train_loader, val_loader, optimizer)
        if accuracy == 100:
            count = count+1
        else:
            count = 0
    
    # Specify a path
    PATH = "state_dict_model.pt"

    # Save
    torch.save(net.state_dict(), PATH)