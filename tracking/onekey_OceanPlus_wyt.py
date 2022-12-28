
import _init_paths
import os
import yaml
import argparse
from os.path import exists
from utils.utils import load_yaml, extract_logs



def main():

    # os.system('python /media/HardDisk/wyt/EATracker/tracking/train_oceanplus_wyt.py')
    os.system('python /media/HardDisk/wyt/EATracker/tracking/test_oceanplus.py')
    os.system('python /media/HardDisk/wyt/EATracker/lib/core/eval_davis.py')



if __name__ == '__main__':
    main()
