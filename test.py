import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import Network

# Specify a path
PATH = "state_dict_model.pt"

# Load
model = Network()
model.load_state_dict(torch.load(PATH))
model.eval()


# LOADING DATA, MIGHT NEED TO PUT THIS IN ITS OWN FILE
numbers = pd.read_csv('data\Results_08072021_400_200.csv', usecols = [i for i in range(1, 21)])

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



targets = np.zeros([results.shape[0], 1])     # if 1 appears in next game, model should output 1, if not then output 0
                                                # use results.shape[0] because we need same number of targets as sets

num_row = 0
num_target = 0
while num_row < results.shape[0]-1:
    targets[num_target][0] = results[num_row + 1][0]
    
    num_row = num_row + 1
    num_target = num_target + 1

#print(targets)

######################################################
# Target splitting for 4 games at a time
target_4 = np.zeros([len(split_4), 1])
num = 0
while num != len(split_4):
    target_4[num][0] = targets[4*(num+1)-1][0]
    num += 1



full_input = torch.from_numpy(train_4)
full_target = torch.from_numpy(target_4)

full_input = full_input.to(torch.float32)
full_target = full_target.to(torch.float32)

input_game = 0
correct = 0.0
while input_game < len(full_input)-1:

    output = model(full_input[input_game])

    pred = (output >= 0.5).float()

    if pred == full_target[input_game]:
        correct += 1
    
    print("Input game: ", full_input[input_game])
    print("Next game", full_input[input_game+1])

    print("Pred: ", pred)
    print("Target: ", full_target[input_game])
    
    input_game += 1

acc = correct/len(full_input)
print("Acc: ", acc)

tot = 0.0
for num in targets:
    if num == 0:
        tot += 1

zero_acc = tot/len(full_input)
print("If just predicting 0: ", zero_acc)