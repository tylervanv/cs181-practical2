# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
from collections import Counter
from datetime import datetime
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import cPickle as pickle
import sys
from scikits.statsmodels.distributions import ECDF # installed from http://scikits.appspot.com/statsmodels

import util

TRAIN_DIR = "train"

call_list = pickle.load(open('tag_list.p', 'r'))
call_set = set(call_list)


#features = ['sleep', 'dump_line']
features = call_list + map(lambda s : s + ' indicator', call_list) + ['bytes_sent', 'bytes_received', 'any_sent', 'any_received', 'totaltime']
#features = call_list + ['bytes_sent', 'bytes_received', 'any_sent', 'any_received', 'totaltime']



#def add_to_set(tree):
#    for el in tree.iter():
#        call = el.tag
#        call_set.add(call)

def create_data_matrix(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = [] 
    i = -1

    #for datafile in os.listdir(direc):
    #    if datafile == '.DS_Store':
    #        continue
    #    tree = ET.parse(os.path.join(direc,datafile))
    #    add_to_set(tree)
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i % 100 == 0:
            print i
            sys.stdout.flush()
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        #for elt in tree.getroot():
        #    print [e.tag=='thread' for e in elt]
        #    #print (elt.tag, elt.attrib, [e[0].tag=='all_section' for e in elt])
        #print
        #print

        this_row_dict = call_feats(tree)
        this_row = np.array([(this_row_dict[feature] if feature in this_row_dict else 0) for feature in features])
        #this_row = call_feats(tree)
        if X is None:
            X = this_row
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

def call_feats(tree):
    #good_calls = ['sleep', 'dump_line']
    #good_calls = call_list + ['bytes_sent', 'bytes_received', 'totaltime']

    call_counter = {}
    call_counter['bytes_sent'] = 0
    call_counter['bytes_received'] = 0
    call_counter['any_sent'] = 0
    call_counter['any_received'] = 0
    call_counter['totaltime'] = 0
    for el in tree.iter():
        call = el.tag
        if call not in call_counter:
            call_counter[call] = 1
            call_counter[call + ' indicator'] = 1
        else:
            call_counter[call] += 1
        if call == 'send_socket':
            call_counter['bytes_sent'] += int(el.attrib['buffer_len'])
            call_counter['any_sent'] = 1
        elif call == 'recv_socket':
            call_counter['bytes_received'] += int(el.attrib['buffer_len'])
            call_counter['any_received'] = 1
        elif call == 'process':
            endtime = datetime.strptime(el.attrib['terminationtime'], '%M:%S.%f')
            starttime = datetime.strptime(el.attrib['starttime'], '%M:%S.%f')
            call_counter['totaltime'] += (endtime - starttime).total_seconds()


    ##### TRY ADDING (BINARY?) FEATURES FOR SOCKET NUMBER -- e.g. sockets 1772, 1812, and 1828 are usually Swizzor, while 2068 is never Swizzor


    #call_feat_array = np.zeros(len(good_calls))
    #for i in range(len(good_calls)):
    #    call = good_calls[i]
    #    call_feat_array[i] = 0
    #    if call in call_counter:
    #        call_feat_array[i] = call_counter[call]

    return call_counter
    #return call_feat_array

## Feature extraction
def main():
    X_train, t_train, train_ids = create_data_matrix(0, 5, TRAIN_DIR)
    X_valid, t_valid, valid_ids = create_data_matrix(10, 15, TRAIN_DIR)

    print 'Data matrix (training set):'
    print X_train
    print train_ids
    print 'Classes (training set):'
    print t_train

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).


    print 'unique tags:', call_set
    print
    print

    X, t, ids = create_data_matrix(0, 10000, TRAIN_DIR)

    X = X.T
    for i in range(len(X)):
        if X[i].any() and not X[i].all():
            X = np.vstack((X, ECDF(X[i])(X[i])))
            features.append(features[i] + ' ECDF')
    X = X.T
    print X.shape

    pickle.dump((ids, X, t, features), open('train_data_new.p', 'w'))

if __name__ == "__main__":
    main()