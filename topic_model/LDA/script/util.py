import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import random
import numpy as np

def read_list(filename):
    with open(filename, 'r') as file:
        l = file.read().split('\n')
    if len(l[-1]) == 0:
        l = l[:-1]
    return l

def write_list(filename, list):
    with open(filename, 'w') as file:
        file.write('\n'.join(list))


def write_corpus(filename, header, list):
    fw = open(filename, 'w')
    fw.write(header + '\n')
    fw.write('\n'.join(list))
    fw.close