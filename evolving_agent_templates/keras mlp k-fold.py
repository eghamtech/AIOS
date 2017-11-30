# AICHOO OS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

#start_of_genes_definitions
#key=fields_to_use;   type=random_int;	from=40;	to=58;	step=1
#key=data;            type=random_array_of_fields;	length=58
#key=nfolds;          type=random_int;	from=2;		to=10;	step=1
#key=optimizer;	      type=random_from_set;			set='TFOptimizer','sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam'
#key=activation;      type=random_from_set;         set='relu','elu','selu','tanh','sigmoid','hard_sigmoid','softplus','softsign','linear'
#key=layers;          type=random_int;	from=2;		to=10;	step=1
#key=neurons;         type=random_int;	from=4;	    to=256;	step=1        
#end_of_genes_definitions

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


import pandas as pd
import math

# obtain a unique ID for the current instance
result_id = {id}
# create new field name based on "field_ev_prefix" (also specified in Constants) with unique instance ID
# and filename to save new field data
field_ev_prefix = "ev_field_"
output_column = field_ev_prefix + str(result_id)
output_filename = output_column + ".csv"

# obtain random field which will be the prediction target of this instance
target_definition = "{source_field}"
# field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
# load these two parts into variables
target_col = target_definition.split("|")[0]
target_file = target_definition.split("|")[1]

# obtain random selection of fields; number of fields to be selected specified in {data}:length gene for this instance
data_defs = {data}

# remove the target field for this instance from the data used for training
if target_definition in data_defs:
    data_defs.remove(target_definition)

#############################################################
#
#                   DATA PREPARATION
#
#############################################################

# read data from the original data file loaded into Memory (specified in Constants as "trainfile")
# "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
main_data = pd.read_csv(workdir+trainfile)
# read data from CSV file containing the prediction target field selected for this instance
df = pd.read_csv(workdir+target_file)[[target_col]]

# read each required field's data from a corresponding CSV file
# number of fields actually read specified in {fields_to_use} gene
cols_count = 0
for i in range(0,len(data_defs)):
    cols_count+=1
    if cols_count>{fields_to_use}:
        break
    col_name = data_defs[i].split("|")[0]
    file_name = data_defs[i].split("|")[1]
    
    if file_name==trainfile:
        df[col_name] = main_data[col_name]
    else:
        # read column from another CSV file and add to df
        df = df.merge(pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)

print ("data loaded", len(df), "rows; ", len(df.columns), "columns")

# analyse target column whether it is binary which may result in different loss function used
is_binary = df.sort_values(target_col)[target_col].unique().tolist()==[0, 1]
if is_binary:
    print ("detected binary target. use LOGLOSS")
    # param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'binary:logistic', 'eval_metric':'logloss', 'nthread':4}
else:
    print ("use MAE")
    # param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'reg:linear', 'eval_metric':'mae', 'nthread':4}


#############################################################
#                   MAIN LOOP
#############################################################
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten
from keras.callbacks      import EarlyStopping, Callback
from sklearn.model_selection import StratifiedKFold

early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto' )
kfolds = {nfolds}



