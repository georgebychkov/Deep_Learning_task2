import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def one_hot_encode(arr, n_labels):
    
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


def get_batches(arr, n_seqs, n_steps):
    
    batch_size = n_seqs * n_steps
    n_batches = len(arr)//batch_size
    
    arr = arr[:n_batches * batch_size]
    
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        
        x = arr[:, n:n+n_steps]
        
        y = np.zeros_like(x)
        
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


def train_model(net, data, epochs=10, n_seqs=10, n_steps=50, opt=lambda x: torch.optim.Adam(x), clip=5, val_frac=0.1, device="cpu"):
    
    optimiser = opt(net.parameters())

    net.train()

    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    net.to(device)
    
    counter = 0
    n_chars = len(net.chars)
    
    train_loss_arr = []
    val_loss_arr = []
    
    for _ in range(epochs):
        
        train_loss_ep = []
        val_loss_ep = []
        
        h = net.init_hidden(n_seqs)
        
        for x, y in get_batches(data, n_seqs, n_steps):
            
            counter += 1
            
            x = one_hot_encode(x, n_chars)
            
            inputs, targets = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

            h = tuple([each.data for each in h])

            net.zero_grad()
            
            output, h = net.forward(inputs, h)
            
            loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.cuda.LongTensor))

            loss.backward()
            
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            optimiser.step()
            
            if counter % 10 == 0:
                
                val_h = net.init_hidden(n_seqs)
                val_losses = []
                
                for x, y in get_batches(val_data, n_seqs, n_steps):
                    
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    inputs, targets = inputs.to(device), targets.to(device)

                    output, val_h = net.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.cuda.LongTensor))
                
                    val_losses.append(val_loss.item())

                train_loss_ep.append(loss.item())
                val_loss_ep.append(np.mean(val_losses))

        train_loss_arr.append(np.mean(train_loss_ep))
        val_loss_arr.append(np.mean(val_loss_ep))

    return train_loss_arr, val_loss_arr


def sample(net, size, prime='The', top_k=None, device="cpu"):
        
    net.to(device)

    net.eval()
    
    chars = [ch for ch in prime]
    
    h = net.init_hidden(1)
    
    for ch in prime:
        char, h = net.predict(ch, h, device=device, top_k=top_k)

    chars.append(char)
    
    for ii in range(size):
        
        char, h = net.predict(chars[-1], h, device=device, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
