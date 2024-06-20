
# new version (change to recursive procedure)
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

def recursive_sampling(layers, layer, order, target, memory):
    # print("current leyer", layer, "memory", len(memory), "target", target)
    if target == 0 or layer<=0:
        return memory
    else:
        nodes = layers[layer-1]
        selected = sample(nodes, order, memory, target)
        memory.extend(selected)
        return recursive_sampling(layers, layer-1, order, target-len(selected), memory)
    
def BallTreeSubsample(data, target_size):
    bt = BallTree(data, leaf_size = 2)
    nodes = bt.get_arrays()[2]
    order = bt.get_arrays()[1]
    layers, n_layers = split_layer(nodes)
    selected = recursive_sampling(layers, n_layers, order, target_size, [])
    return selected
    