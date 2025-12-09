import torch
import random
import numpy as np
import os
import datetime


def log(message: str, 
        file: str = 'log_file.txt', 
        ):
    now = datetime.datetime.now()
    with open(file, 'a') as out_file:
        out_file.write(f'{now}\t{message}')
        out_file.write('\n')


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def get_file_paths(file_path: str = 'MetalSurfaceDefectsData/train'):
    dirs = [file_path]
    files = {}
    while len(dirs) > 0:
        d = dirs[0]
        for f in os.listdir(d):
            name = f'{d}/{f}'
            if os.path.isfile(name):
                if not d in files:
                    files[d] = []
                files[d].append(name)
            else:
                dirs.append(name)
        dirs.remove(d)
    for key in files.keys():
        files[key].sort()
    return {f: files[f] for f in sorted(files)}  
