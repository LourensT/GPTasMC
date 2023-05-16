# Use shakespeare corpus, 
# reduce to three tokens (space, vowel, consonant), and consider context length 4
# Statespace has size 81.

# %%
import itertools
import numpy as np

import torch 
from torch.nn import functional as F

from babyGPT import GPT, GPTConfig
# seed for reproducibility
torch.manual_seed(0)

batch_size = 4 # how many independent sequences will we process in parallel?
context_length = 4 # what is the maximum context length for predictions?
vocab_size = 3 # space, vowel, consonant

def text_to_consonants_and_values(text):
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'

    # remove capitalization
    text = text.lower()

    tokenized = []
    for c in text:
        if c in vowels:
            tokenized.append(1)
        elif c in consonants:
            tokenized.append(2)
        else: # make space 
            tokenized.append(0)
    
    return torch.tensor(tokenized, dtype=torch.long)

def get_batch():
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+context_length] for i in ix])
    return x, y

# %%
if __name__=="__main__":
    with open('./data/full-shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    data = text_to_consonants_and_values(text)

    # configure the model hyperparameters
    config = GPTConfig(
        block_size = context_length,
        vocab_size = vocab_size,
        n_layer = 4,
        n_head = 4,
        n_embd = 64,
        bias = False,
    )
    # instanstiate model
    m = GPT(config)
    # set optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # %%
    eval_interval = 1000
    max_iters = 20000
    losses = []
    for i in range(max_iters): # increase number of steps for good results... 
    
        # sample a batch of data
        xb, yb = get_batch()

        # evaluate the loss
        logits = m(xb)
        loss = F.cross_entropy(logits, yb) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # every once in a while evaluate the losses on last 100 batches
        if i % eval_interval == 0 or i == max_iters - 1:
            print(i, np.mean(losses[-eval_interval:]))
    
        losses.append(loss.item())

    print(f"final loss {loss.item()}")
    # %%

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long)
    print(''.join(m.generate(context, max_new_tokens=100)[0].tolist()))

    # %%
    # plot loss over time with rolling average
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.plot(pd.Series(losses).rolling(eval_interval).mean())
    plt.title("Loss over time (smoothed)")
    plt.xlabel("Iterations")
    plt.ylabel("Cross entropy loss")
    plt.show()

    # %%
    # get the transition matrix
    # get all possible states
    # state to index
    states = list(itertools.product((0, 1, 2), repeat=context_length))
    statesize = len(states)
    state_to_index = { s:i for i,s in enumerate(states)}

    # calculate transition matrix in logits
    transition = np.zeros((statesize, statesize))
    for i, state in enumerate(states):
        logits = m(torch.tensor(state, dtype=torch.long)[None, ...])
        for i, rho in enumerate(logits[0]):
            new_state = state[1:] + (i,)
            transition[state_to_index[state], state_to_index[new_state]] = rho.item()

    # save transition matrix to a csv
    with open(f"logits_vocab{vocab_size}context{context_length}.csv", "w") as f:
        # write states as header
        for i, state in enumerate(states):
            if i > 0:
                f.write(",")
            f.write("".join(map(str, state)))
        f.write("\n")

        # write logits
        for row in transition:
            for i, cell in enumerate(row):
                if i > 0:
                    f.write(",")
                f.write(str(cell.item()))
            f.write("\n")

# %%
