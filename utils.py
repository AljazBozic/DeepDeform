import os
import shutil
import numpy as np
import struct


def load_flow(filename):
    # Flow is stored row-wise in order [channels, height, width].
    assert os.path.isfile(filename), "File not found: {}".format(filename)

    flow = None
    with open(filename, 'rb') as fin:
        width = struct.unpack('I', fin.read(4))[0]
        height = struct.unpack('I', fin.read(4))[0]
        channels = struct.unpack('I', fin.read(4))[0]
        n_elems = height * width * channels

        flow = struct.unpack('f' * n_elems, fin.read(n_elems * 4))
        flow = np.asarray(flow, dtype=np.float32).reshape([channels, height, width])

    return flow
    

def save_flow(filename, flow_input):
    flow = np.copy(flow_input)

    # Flow is stored row-wise in order [channels, height, width].
    assert len(flow.shape) == 3
    
    with open(filename, 'wb') as fout:
        fout.write(struct.pack('I', flow.shape[2]))
        fout.write(struct.pack('I', flow.shape[1]))
        fout.write(struct.pack('I', flow.shape[0]))
        fout.write(struct.pack('={}f'.format(flow.size), *flow.flatten("C")))