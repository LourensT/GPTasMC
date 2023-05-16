# %%
# Description: This file contains the code for the diagram of the model
from graphviz import Digraph
import torch
import numpy as np

# %%

# filepath to csv holding logits
logits_FP = "logits_vocab2context3asdf.csv"
temparature = 0.5

# read logits from csv
with open(logits_FP) as f:
    lines = f.readlines()
    states = lines[0].strip().split(",")
    transitions = []
    for line in lines[1:]:
        transitions.append(list(map(float, line.strip().split(","))))

# %%
import pickle
with open("./logits/logits_vocab3context4.pkl", "rb") as f:
    transitions = pickle.load(f)

import itertools
states = [''.join(n) for n in itertools.product("012", repeat=4)]

# %%
# convert each row of logits to probabilities
for i in range(len(transitions)):
    row = transitions[i]
    row = [x/temparature if x != 0.0 else -np.infty for x in row]
    row = torch.softmax(torch.tensor(row), dim=0)
    transitions[i] = row.tolist()

# create graph
dot = Digraph(comment='logits_FP', engine='circo')
for i in range(len(states)):
    dot.node(str(i), states[i])
for i in range(len(transitions)):
    for j in range(len(transitions[i])):
        if transitions[i][j] > 0.01:
            dot.edge(str(i), str(j), label=str(round(transitions[i][j], 2)))
