import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import cPickle as pickle

import util

TRAIN_DIR = "train"

call_set = set([])

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

for datafile in os.listdir(TRAIN_DIR):
    if datafile == '.DS_Store':
        continue
    tree = ET.parse(os.path.join(TRAIN_DIR, datafile))
    for e in tree.iter():
        call_set.add(e.tag)
    add_to_set(tree)

pickle.dump(list(call_set), open('tag_list.p', 'w'))
print len(call_set)
print call_set