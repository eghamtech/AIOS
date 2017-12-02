#start_of_genes_definitions
#key=fields_to_use;  type=random_int;  from=5;	to=60;	step=1
#key=data;            type=random_array_of_fields;	length=60
#key=folds;           type=random_int;	from=10;	to=10;	step=1
#key=optimizer;	      type=random_from_set;         set='sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam'
#key=activation;      type=random_from_set;         set='relu','elu','selu','tanh','sigmoid','hard_sigmoid','softplus','softsign','softmax','linear'
#key=activation_output;      type=random_from_set;         set='relu','elu','selu','tanh','sigmoid','hard_sigmoid','softplus','softsign','softmax','linear'
#key=layers;          type=random_int;	from=2;		to=10;	step=1
#key=neurons;         type=random_int;	from=4;	    to=256;	step=1   
#key=batch_size;      type=random_int;	from=5;	    to=256;	step=1
#key=epochs;          type=random_int;	from=5;	    to=100; step=1
#key=dropout;         type=random_float;	from=0.02;	to=0.7;	step=0.02
#end_of_genes_definitions

# AICHOO OS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
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
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, log_device_placement=True, allow_soft_placement=True)
# allocate only as much GPU memory as needed by runtime - otherwise all GPU memory is reserved and mutiple processes cannot use GPU 
session_conf.gpu_options.allow_growth = True

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

# obtain random field (same for all instances within the evolution) which will be the prediction target for this instance/evolution
target_definition = "{field_to_predict}"
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
#                   DATA PREPARATION
#############################################################

# read data from the original data file loaded into Memory (specified in Constants as "trainfile")
# "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
main_data = pd.read_csv(workdir+trainfile)
# read data from CSV file containing the prediction target field selected for this instance
df = pd.read_csv(workdir+target_file)[[target_col]]

# read each required field's data from a corresponding CSV file
# number of fields actually read specified in {fields_to_use} gene
n_fields_to_use = {fields_to_use}
cols_count = 0
for i in range(0,len(data_defs)):
    cols_count+=1
    if cols_count>n_fields_to_use:
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
    print ("detected binary target; use Binary Cross Entropy loss evaluation")
    s_loss_function = 'binary_crossentropy'
    n_classes = 1
else:
    print ("detected non-binary target; use MSE loss evaluation")
    s_loss_function = 'mean_squared_error'
    n_classes = 1

#############################################################
#                   MLP Model Compiling
#############################################################
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten
from keras.callbacks      import EarlyStopping, Callback
#from sklearn.model_selection import StratifiedKFold, KFold

n_folds = {folds}
s_optimizer = {optimizer}
s_activation = {activation}
n_layers = {layers}
n_neurons = {neurons}
n_batch_size = {batch_size}
n_epochs = {epochs}
n_dropout = {dropout}
s_output_activation = {activation_output}

early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto' )
mlp_model = Sequential()

# add hidden layers 
for i in range(n_layers):
    if i == 0:
        mlp_model.add(Dense(n_neurons, activation=s_activation, input_dim=n_fields_to_use))
    else:
        mlp_model.add(Dense(n_neurons, activation=s_activation))

    mlp_model.add(Dropout(n_dropout))

# add output layer
mlp_model.add(Dense(n_classes, activation=s_output_activation))

mlp_model.compile(loss=s_loss_function, optimizer=s_optimizer, metrics=['accuracy'])

#############################################################
#                   MAIN LOOP
#############################################################
block = int(len(df)/n_folds)

prediction = []

weighted_result = 0
count_records_notnull = 0

for fold in range(0,n_folds):
    print ("\nFOLD", fold, "\n")
    range_start = fold*block
    range_end = (fold+1)*block
    if fold==n_folds-1:
        range_end = len(df)
    range_predict = range(range_start, range_end)
    print ("Fold to predict start", range_start, "; end ", range_end)
    
    x_test = df[df.index.isin(range_predict)]
    x_test.reset_index(drop=True, inplace=True)
    x_test_orig = x_test.copy()
    x_test = x_test[x_test[target_col].notnull()]
    x_test.reset_index(drop=True, inplace=True)

    x_train = df[df.index.isin(range_predict)==False]
    x_train.reset_index(drop=True, inplace=True)
    x_train= x_train[x_train[target_col].notnull()]
    x_train.reset_index(drop=True, inplace=True)

    print ("x_test rows count: " + str(len(x_test)))
    print ("x_train rows count: " + str(len(x_train)))

    y_train = np.array( x_train[target_col] )
    x_train = np.array( x_train.drop(target_col, 1) )

    y_test = np.array( x_test[target_col] )
    x_test = np.array( x_test.drop(target_col, 1) )

    mlp_history = mlp_model.fit( x_train, y_train,
                   batch_size=n_batch_size,
                   epochs=n_epochs,  
                   verbose=0,
                   validation_data=(x_test, y_test),
                   callbacks=[early_stopper] )
    
    print(pd.DataFrame(mlp_history.history))
    
    score = mlp_model.evaluate(x_test, y_test, verbose=0)
    print('Test fold loss:', score[0])
    print('Test fold accuracy:', score[1])
    
    result = score[0]
    weighted_result += result * len(x_test)
    count_records_notnull += len(x_test)
    
    if np.isnan(score[0]) or score[1] == 0:
        print ("fitness=99999")
        quit()
    
    pred_all_test = mlp_model.predict(np.array(x_test_orig.drop(target_col, axis=1)), verbose=0)
    pred_all_test = [item for sublist in pred_all_test for item in sublist]
    
    prediction = np.concatenate([prediction,pred_all_test])

weighted_result = weighted_result/count_records_notnull
print ("weighted_result:", weighted_result)

#############################################################
#                   OUTPUT
#############################################################
if output_mode==1:
    df[output_column] = prediction
    df[[output_column]].to_csv(workdir+output_filename)

    print ("#add_field:"+output_column+",N,"+output_filename)
else:
    print ("fitness="+str(weighted_result))

