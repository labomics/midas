# Developing ... 
from sklearn.neighbors import BallTree
import random
def split_layer(nodes):
    layers = {}
    l = -1
    for n in nodes:
        if n[0] == 0:
            layer = []
            l += 1
        layer.append(n)
        layers[l] = layer  
    return layers, l+1
    
def randomsampling(alist, number):
    random.shuffle(alist)
    return alist[:number]

def sample(nodes, order, memory, target):
    nodes_id = []
    for n in nodes:
        ns = list(set(order[n[0]:n[1]])-set(memory))
        nodes_id.append(ns)

    number = target//len(nodes)
    selected = []
    for n in nodes_id:
        selected.extend(randomsampling(n, number))
    return selected
def BallTreeSubsample(data, target):
    bt = BallTree(data, leaf_size = 2)
    nodes = bt.get_arrays()[2]
    order = bt.get_arrays()[1]
    layers, n_layers = split_layer(nodes)
    selected = []
    l = n_layers-1
    while len(selected) < target:
        selected.extend(sample(layers[l], order, selected, target- len(selected)))
        l -= 1
    return selected