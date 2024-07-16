

import numpy as np
import json

from ao_core import ao_core as ao
from ao_core import Arch as aoArch


description = "3600-neuron Agent for ARC benchmarking"      #    MNIST is in grayscale, which we downscaled to B&W for the simple 28x28 neuron count -- 788 = 28x28 + 4
arch_i = [30*30 * 4]               # note that the 784 I neurons are in 1 input channel; MNIST is like a single channel clam, so it's limitations are obvious from the prespective of our approach, more on this here: 
arch_z = [30*30 * 4]                   # 4 neurons in 1 channel as 4 binary digits encodes up to integer 16, and only 10 (0-9) are needed for MNIST
arch_c = []
connector_function = "rand_conn"
connector_parameters = [1200, 1200, 1200, 1200]
# so all Q neurons are connected randomly to 1200 I and 1200 neighbor Q
# and all Z neurons are connected randomly to 784 Q and 4 (or all) neighbor Z

# To maintain compatability with our API, do not change the variable name "Arch" or the constructor class "ao.Arch" in the line below (the API is pre-loaded with a version of the Arch class in this repo's main branch, hence "ao.Arch")
arcArch = aoArch.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description)

arcAgent = ao.Agent( arcArch )


test_file = open("ARC-AGI\data\training\0d3d703e.json")

test_data = json.load(test_file)


def pad_ARC( arr, pad_value=10, final_size=(30,30)):
    # transform to binary
    inpextra = ( (final_size[0]-arr.shape[0])/2, (final_size[1]-arr.shape[1])/2 )

    inpad = [ [0, 0], [0, 0] ]
    irc = 0
    for rc in inpextra:
        if rc % 2 == 0: 
            # if even
            inpad[irc][0] = rc
            inpad[irc][1] = rc
        else:
            # if odd
            inpad[irc][0] = rc - 0.5
            inpad[irc][1] = rc + 0.5
        irc += 1
    inpad = np.asarray(inpad, dtype=int)
    arr_padded = np.pad( arr, inpad, mode='constant', constant_values=pad_value)

    return arr_padded


color_to_binary = [
    '1111', # black
    format(1, '04b'), # blue
    format(2, '04b'), # red
    format(3, '04b'), # green
    format(4, '04b'), # yellow
    format(5, '04b'), # grey
    format(6, '04b'), # pink
    format(7, '04b'), # orange
    format(8, '04b'), # l blue
    format(9, '04b'), # maroon
    '0000', # void / null
]


def ARC_to_binary( input_padded):

    input_flat = input_padded.flatten()
    inn_stringvec = ""
    for p in input_flat:
        inn_stringvec += color_to_binary[p]
    
    inn_narray = np.asarray(list(inn_stringvec), dtype=int)

    return inn_narray



for pair in test_data['train']:

    inp = np.asarray(pair['input'])
    inp_padded = pad_ARC( inp )
    inp_binary = ARC_to_binary(inp_padded)

    onp = np.asarray(pair['output'])
    onp_padded = pad_ARC( onp )
    onp_binary = ARC_to_binary(onp_padded)

    arcAgent.next_state(inp_binary, LABEL=onp_binary )


test_len = len(test_data['test'])

accuracy = np.zeros(test_len)

i = 0
for pair in test_data['test']:

    inp = np.asarray(pair['input'])
    inp_padded = pad_ARC( inp )
    inp_binary = ARC_to_binary(inp_padded)

    for run in range(5):
        arcAgent.next_state(inp_binary)
    z_index = arcAgent.arch.Z__flat
    s = arcAgent.state - 1
    response = arcAgent.story[ s, z_index]

    onp = np.asarray(pair['output'])
    onp_padded = pad_ARC( onp )
    onp_binary = ARC_to_binary(onp_padded)

    accuracy[i] = sum(sum(onp_binary == response)) / (900*4)
    i += 1





x = 0
for a in range(5):
    print(x)
    x += 2
