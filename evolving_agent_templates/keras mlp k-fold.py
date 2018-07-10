#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=800
#key=fields_to_use;  type=random_int;  from=200;  to=800;  step=1
#key=nfolds;  type=random_int;  from=3;  to=3;  step=1
#key=use_validation_set;  type=random_from_set;  set=True
#key=filter_column;  type=random_from_set;  set=Submission_Date_TS
#key=train_set_from;  type=random_from_set;  set=self.timestamp('2013-11-01')
#key=train_set_to;  type=random_from_set;  set=self.timestamp('2014-11-01')
#key=valid_set_from;  type=random_from_set;  set=self.timestamp('2014-11-01')
#key=valid_set_to;  type=random_from_set;  set=self.timestamp('2016-11-01')
#key=filter_column_2;  type=random_from_set;  set=
#key=train_set_from_2;  type=random_from_set;  set=
#key=train_set_to_2;  type=random_from_set;  set=
#key=valid_set_from_2;  type=random_from_set;  set=
#key=valid_set_to_2;  type=random_from_set;  set=
#key=ignore_columns_containing;  type=random_from_set;  set=ev_field
#key=include_columns_containing;  type=random_from_set;  set=scaled_
#key=optimizer;  type=random_from_set;  set='sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam'
#key=activation;  type=random_from_set;  set='relu','elu','selu','tanh','sigmoid','hard_sigmoid','softplus','softsign','softmax','linear'
#key=activation_output;  type=random_from_set;  set='relu','elu','selu','tanh','sigmoid','hard_sigmoid','softplus','softsign','softmax','linear'
#key=layers;  type=random_int;  from=2;  to=10;  step=1
#key=neurons;  type=random_int;  from=4;  to=256;  step=1
#key=batch_size;  type=random_int;  from=5;  to=256;  step=1
#key=epochs;  type=random_int;  from=5;  to=100;  step=1
#key=early_stopping_min_delta; type=random_float;  from=0.01;  to=0.04;  step=0.01
#key=dropout;  type=random_float;  from=0.02;  to=0.7;  step=0.02
#key=num_threads;  type=random_int;  from=4;  to=4;  step=1
#key=use_float32_dtype; type=random_from_set;  set=True
#end_of_genes_definitions

# AICHOO OS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

class cls_ev_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import tensorflow as tf
    import numpy as np
    import random as rn
    import dateutil
    import calendar
    import os.path

    # specify whether to run Keras/TensorFlow on CPU or GPU (and which GPU, if you have multiple)
    s_tf_device = '/cpu:0'
    #s_tf_device = '/gpu:0'
    #s_tf_device = '/gpu:1'
    
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
    
    import pandas as pd
    import math

    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "field_ev_prefix" (also specified in Constants) with unique instance ID
    # and filename to save new field data
    field_ev_prefix = "ev_field_mlp_"
    output_column = field_ev_prefix + str(result_id)
    output_filename = output_column + ".csv"

    # obtain random field (same for all instances within the evolution) which will be the prediction target for this instance/evolution
    target_definition = "{field_to_predict}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    target_col = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]

    # obtain random selection of fields; number of fields to be selected specified in (data):length gene for this instance
    data_defs = {data}
    fields_to_use = {fields_to_use}
    start_fold = {start_fold}
    nfolds = {nfolds}
    num_threads = {num_threads}
    
    # if filter columns are specified then training and validation sets will be selected based on filter criteria
    # based on filter criteria training + validation sets will not necessarily constitute all data, the remainder will be called "test set"
    filter_column = "{filter_column}"
    filter_column_2 = "{filter_column_2}"
    # filter_filename = trainfile           # filter columns are in trainfile which must be specified in Constants - deprecated
    
    # fields matching the specified string will not be used in the model
    ignore_columns_containing = "{ignore_columns_containing}"
    # include only fields matching string e.g., only properly scaled columns should be used with MLP
    include_columns_containing = "{include_columns_containing}"
    
    with tf.device(s_tf_device):
        # Force TensorFlow to use single thread.
        # Multiple threads are a potential source of
        # non-reproducible results.
        # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=num_threads, inter_op_parallelism_threads=num_threads, log_device_placement=True, allow_soft_placement=True)
        # allocate only as much GPU memory as needed by runtime - otherwise all GPU memory is reserved and mutiple processes cannot use GPU 
        session_conf.gpu_options.allow_growth = True

        from keras import backend as K

        # The below tf.set_random_seed() will make random number generation
        # in the TensorFlow backend have a well-defined initial state.
        # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
        tf.set_random_seed(1234)

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        init = tf.global_variables_initializer()
        sess.run(init)
        K.set_session(sess)
    
    def is_set(self, s):
        return len(s)>0 and s!="0"
    
    def is_use_column(self, s):
        # determine whether given column should be used or ignored
        if s.find(self.target_col)>=0:  # ignore columns that contain target_col as they are a derivative of the target
            return False
        # ignore other columns containing specified parameter value
        if self.is_set(self.ignore_columns_containing) and s.find(self.ignore_columns_containing)>=0:
            return False
        # include all columns if include parameter not specified
        if not self.is_set(self.include_columns_containing):
            return True
        # include columns specified in parameter
        if self.is_set(self.include_columns_containing) and s.find(self.include_columns_containing)>=0:
            return True
        # ignore all other columns
        return False
   
    def timestamp(self, x):
        return self.calendar.timegm(self.dateutil.parser.parse(x).timetuple())
 
    def my_log_loss(self, a, b):
        eps = 1e-9
        sum1 = 0.0
        for k in range(0, len(a)):
            bx = min(max(b[k],eps), 1-eps)
            sum1 += 1.0 * a[k] * self.math.log(bx) + 1.0 * (1 - a[k]) * self.math.log(1 - bx)
        return -sum1/len(a)
    
    
    def __init__(self):
        # remove the target field for this instance from the data used for training
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)
        
        with self.tf.device(self.s_tf_device):
            from keras.models import load_model
            # if saved model for the target field already exists then load it from filesystem
            self.predictors = []
            for fold in range(self.start_fold, self.nfolds):
                if self.os.path.isfile(workdir + self.output_column + "_fold" + str(fold) + ".model"):               
                    predictor_stored = load_model(workdir + self.output_column + "_fold" + str(fold) + ".model")
                    self.predictors.append(predictor_stored)
                
        # obtain columns definitions to filter data set by
        if self.is_set(self.filter_column):
            self.filter_filename = self.filter_column.split("|")[1]
            self.filter_column = self.filter_column.split("|")[0]
      
        if self.is_set(self.filter_column_2):
            self.filter_filename_2 = self.filter_column_2.split("|")[1]
            self.filter_column_2 = self.filter_column_2.split("|")[0]
    
    
    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        columns_new = []
        columns = []
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        cols_count = 0
        for i in range(0,len(self.data_defs)):
            col_name = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]
            
            if self.is_use_column(col_name):
                cols_count+=1
                if cols_count > self.fields_to_use:
                    break
                # assemble dataframe column by column
                if cols_count==1:
                    df = df_add[[col_name]]
                else:
                    df = df.merge(df_add[[col_name]], left_index=True, right_index=True)
                
                # df[col_name] = df[col_name].fillna(0) # replace NaN in each column with 0 as this is crucial for Keras
                # above doesn't work for duplicate columns in DF but all columns must be pre-scaled anyway without NaN
                
                # some columns may appear multiple times in data_defs as inhereted from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                columns.append(col_name)
                ncol_count = columns.count(col_name)
                if ncol_count==1:
                    columns_new.append(col_name)
                else:
                    columns_new.append(col_name+"_v"+str(ncol_count))
        
        # rename columns in df to unique names
        df.columns = columns_new      
   
        # apply previously loaded model to new data and obtain predictions
        # predict new data set in df applying model for each fold used for training
        with self.tf.device(self.s_tf_device):
            pred = self.np.zeros(len(df))
            for fold in range(self.start_fold, self.nfolds):
                pred += self.predictors[fold-self.start_fold].predict(self.np.array(df), verbose=0)
            # average prediction over all folds    
            pred = pred / (self.nfolds - self.start_fold)
            df_add[self.output_column] = pred
        
    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        # global trainfile
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        from datetime import datetime
        print ("enter run mode " + str(mode))  # 0=work for fitness only;  1=make new output field
        
        use_validation_set = {use_validation_set}
        use_float32_dtype = {use_float32_dtype}
        
        # obtain indexes for train, validation and remainder sets, if validation set is required
        if use_validation_set:
            df_filter_column = self.pd.read_csv(workdir+self.filter_filename, usecols = [self.filter_column])
            if self.is_set(self.filter_column_2):
                df_t = self.pd.read_csv(workdir+self.filter_filename_2, usecols = [self.filter_column_2])
                df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)
                
            if not self.is_set(self.filter_column_2):
                # one filter column used
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>={train_set_from}, df_filter_column[self.filter_column]<{train_set_to})
                train_indexes = df_filter_column[condition1].index
                test_indexes = df_filter_column[self.np.logical_not(condition1)].index
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>={valid_set_from}, df_filter_column[self.filter_column]<{valid_set_to})
                validation_set_indexes = df_filter_column[condition1].index
            else:
                # two filter columns specified
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>={train_set_from}, df_filter_column[self.filter_column]<{train_set_to})
                condition2 = self.np.logical_and(df_filter_column[self.filter_column_2]>={train_set_from_2}, df_filter_column[self.filter_column_2]<{train_set_to_2})
                train_indexes = df_filter_column[self.np.logical_and(condition1, condition2)].index
                test_indexes = df_filter_column[self.np.logical_not(self.np.logical_and(condition1, condition2))].index
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>={valid_set_from}, df_filter_column[self.filter_column]<{valid_set_to})
                condition2 = self.np.logical_and(df_filter_column[self.filter_column_2]>={valid_set_from_2}, df_filter_column[self.filter_column_2]<{valid_set_to_2})
                validation_set_indexes = df_filter_column[self.np.logical_and(condition1, condition2)].index
            
            print ("Length of train set:", len(train_indexes))
            print ("Length of test/remainder set:", len(test_indexes))
            print ("Length of validation set:", len(validation_set_indexes))
            
        #############################################################
        #                   DATA PREPARATION
        #############################################################
        
        # "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
        # read data from CSV file containing the prediction target field selected for this instance
        df = self.pd.read_csv(workdir+self.target_file, usecols=[self.target_col])[[self.target_col]]

        columns_new = [self.target_col]
        columns = [self.target_col]
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        print (str(datetime.now()), " start loading data")
        cols_count = 0
        block_progress = 0
        block = int(self.fields_to_use/20)
        
        for i in range(0,len(self.data_defs)):
            col_name = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]
            
            if self.is_use_column(col_name):
                cols_count+=1
                if cols_count > self.fields_to_use:
                    break
   
                df_col = self.pd.read_csv(workdir+file_name, usecols=[col_name])[[col_name]]  # read column from csv file
                if df_col[col_name].dtype == self.np.float64 and use_float32_dtype:           # downcast to save memory if needed
                    df_col[col_name] = df_col[col_name].astype(self.np.float32)
                    
                df = df.merge(df_col, left_index=True, right_index=True)
                
                block_progress += 1
                if (block_progress >= block):
                    block_progress = 0
                    print (str(datetime.now()), " data loaded: ", round(cols_count/self.fields_to_use*100,2), "%")
                    
                # some columns may appear multiple times in data_defs as inhereted from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                columns.append(col_name)
                ncol_count = columns.count(col_name)
                if ncol_count==1:
                    columns_new.append(col_name)
                else:
                    columns_new.append(col_name+"_v"+str(ncol_count))
                
                # df[col_name] = df[col_name].fillna(0) # replace NaN in each column with 0 as this is crucial for Keras 
                # above doesn't work for duplicate columns in DF but all columns must be pre-scaled anyway without NaN
               
        # rename columns in df to unique names
        df.columns = columns_new
        data_fields_count = len(df.columns)-1  # need this for building MLP model layers; df.columns includes the target column, hence need to do -1
        print (str(datetime.now()), " data loaded", len(df), "rows; ", len(df.columns), "columns")
        #print ("Columns used: ", columns_new)
        
        original_row_count = len(df)
        
        if use_validation_set:
            # use previously calculated indexes to select train, validation and remainder sets
            df_test = df[df.index.isin(test_indexes)]
            df_test.reset_index(drop=True, inplace=True)
            self.df_valid = df[df.index.isin(validation_set_indexes)]
            self.df_valid.reset_index(drop=True, inplace=True)
            df = df[df.index.isin(train_indexes)]
            df.reset_index(drop=True, inplace=True)
            # initialise prediction columns for validation and test as they will be aggregate predictions from multiple folds
            predicted_valid_set = self.np.zeros(len(self.df_valid))
            predicted_test_set = self.np.zeros(len(df_test))
            
        # analyse target column whether it is binary which may result in different loss function used
        is_binary = df[df[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()==[0, 1]
        if is_binary:
            print ("detected binary target; use Binary Cross Entropy loss evaluation")
            s_loss_function = 'binary_crossentropy'
            s_metrics = 'accuracy'
            n_classes = 1
        else:
            print ("detected non-binary target; use MSE loss evaluation")
            s_loss_function = 'mean_squared_error'
            s_metrics = 'mean_squared_error'
            n_classes = 1

        s_optimizer = {optimizer}
        s_activation = {activation}
        n_layers = {layers}
        n_neurons = {neurons}
        n_batch_size = {batch_size}
        n_epochs = {epochs}
        n_dropout = {dropout}
        s_output_activation = {activation_output}
        n_early_stopping_min_delta = {early_stopping_min_delta}

        #############################################################
        #                   MLP Model Compiling
        #############################################################
        from keras.models         import Sequential
        from keras.layers         import Dense, Dropout
        from keras.callbacks      import EarlyStopping

        with self.tf.device(self.s_tf_device):
            early_stopper = EarlyStopping( monitor='val_loss', min_delta=n_early_stopping_min_delta, patience=2, verbose=0, mode='auto' )
            mlp_model = Sequential()

            # add hidden layers 
            for i in range(n_layers):
                if i == 0:
                    mlp_model.add(Dense(n_neurons, activation=s_activation, input_dim=data_fields_count))
                else:
                    mlp_model.add(Dense(n_neurons, activation=s_activation))

                mlp_model.add(Dropout(n_dropout))

            # add output layer
            mlp_model.add(Dense(n_classes, activation=s_output_activation))

            mlp_model.compile(loss=s_loss_function, optimizer=s_optimizer, metrics=[s_metrics])

            #############################################################
            #                   MAIN LOOP
            #############################################################
            # divide training data into nfolds of size block
            block = int(len(df)/self.nfolds)

            prediction = self.np.zeros(len(df))

            weighted_result = 0
            weighted_auc = 0
            count_records_notnull = 0

            for fold in range(self.start_fold, self.nfolds):
                print ("\nFOLD", fold, "\n")
                range_start = fold*block
                range_end = (fold+1)*block
                if fold==self.nfolds-1:
                    range_end = len(df)
                range_predict = range(range_start, range_end)
                print ("Fold to predict start", range_start, "; end ", range_end)

                x_test = df[df.index.isin(range_predict)]
                x_test.reset_index(drop=True, inplace=True)
                x_test_orig = x_test.copy()                                 # save original test set before removing null values
                x_test = x_test[x_test[self.target_col].notnull()]          # remove examples that have no proper target label
                x_test.reset_index(drop=True, inplace=True)

                x_train = df[df.index.isin(range_predict)==False]
                x_train.reset_index(drop=True, inplace=True)
                x_train = x_train[x_train[self.target_col].notnull()]       # remove examples that have no proper target label
                x_train.reset_index(drop=True, inplace=True)

                print ("x_test rows count: " + str(len(x_test)))
                print ("x_train rows count: " + str(len(x_train)))

                y_train = self.np.array( x_train[self.target_col] )          # separate training fields and the target
                x_train = self.np.array( x_train.drop(self.target_col, 1) )

                y_test = self.np.array( x_test[self.target_col] )
                x_test = self.np.array( x_test.drop(self.target_col, 1) )

                mlp_history = mlp_model.fit( x_train, y_train,
                                             batch_size=n_batch_size,
                                             epochs=n_epochs,  
                                             verbose=0,
                                             validation_data=(x_test, y_test),
                                             callbacks=[early_stopper] )

                print(self.pd.DataFrame(mlp_history.history))

                score = mlp_model.evaluate(x_test, y_test, verbose=0)
                print('Test fold loss:', score[0])
                print('Test fold accuracy:', score[1])

                if mode==1:
                    mlp_model.save(workdir + self.output_column + "_fold" + str(fold) + ".model")

                if self.np.isnan(score[0]) or score[1] == 0:
                    print ('Test Loss is NaN or Accuracy = 0, no point to carry on with more folds')
                    weighted_result = 99999*count_records_notnull      
                    break
                
                pred = mlp_model.predict(x_test, verbose=0)
                if is_binary:
                    result = score[0]
                    # show various metrics as per
                    # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                    result_roc_auc = roc_auc_score(y_test, pred)
                    result_cm = confusion_matrix(y_test, (pred>0.5))  # assume 0.5 probability threshold
                    result_cr = classification_report(y_test, (pred>0.5))
                    print ("ROC AUC score: ", result_roc_auc)
                    print ("Confusion Matrix:\n", result_cm)
                    print ("Classification Report:\n", result_cr)
                else:
                    result = mean_squared_error(y_test, pred)
                    
                weighted_result += result * len(x_test)
                weighted_auc += result_roc_auc * len(pred)
                count_records_notnull += len(x_test)
                
                # predict all examples in the original test set which may include erroneous examples previously removed
                pred_all_test = mlp_model.predict(self.np.array(x_test_orig.drop(self.target_col, axis=1)), verbose=0)
                pred_all_test = [item for sublist in pred_all_test for item in sublist]

                # prediction = self.np.concatenate([prediction,pred_all_test])
                prediction[range_start:range_end] = pred_all_test
                
                # predict validation and remainder sets examples
                if use_validation_set:
                    pred1 = mlp_model.predict(self.np.array(self.df_valid.drop(self.target_col, axis=1)), verbose=0)
                    pred1 = [item for sublist in pred1 for item in sublist]
                    predicted_valid_set += self.np.array(pred1)
                    
                    pred2 = mlp_model.predict(self.np.array(df_test.drop(self.target_col, axis=1)), verbose=0)
                    pred2 = [item for sublist in pred2 for item in sublist]
                    predicted_test_set += self.np.array(pred2)

            weighted_result = weighted_result/count_records_notnull
            weighted_auc = weighted_auc/count_records_notnull
            print ("weighted_result:", weighted_result)
            print ("weighted_auc:", weighted_auc)

            if use_validation_set:
                print()
                print()
                print ("*************  VALIDATION SET RESULTS  *****************")
                print ("Length of validation set:", len(predicted_valid_set))
                predicted_valid_set = predicted_valid_set / (self.nfolds - self.start_fold)
                predicted_test_set = predicted_test_set / (self.nfolds - self.start_fold)

                # validation set may have missing labels (NAN), for metrics calc find subset with proper labels
                self.df_valid['predicted_valid_set'] = predicted_valid_set
                self.df_valid = self.df_valid[self.df_valid[self.target_col].notnull()]
                self.df_valid.reset_index(drop=True, inplace=True)
                y_valid = self.df_valid[self.target_col]
                predicted_valid_set = self.df_valid['predicted_valid_set']
                
                if is_binary:                 
                    try:
                        result = self.my_log_loss(y_valid, predicted_valid_set)
                        print ("LOGLOSS: ", result)
                        result_roc_auc = roc_auc_score(y_valid, predicted_valid_set)
                        print ("ROC AUC score: ", result_roc_auc)
                        result_cm = confusion_matrix(y_valid, (predicted_valid_set>0.5))  # assume 0.5 probability threshold
                        print ("Confusion Matrix:\n", result_cm)
                        result_cr = classification_report(y_valid, (predicted_valid_set>0.5))
                        print ("Classification Report:\n", result_cr)
                    except Exception as e:
                        print (e)
                else:
                    #result = sum(abs(y_valid-predicted_valid_set))/len(y_valid)
                    #print ("MAE: ", result)
                    result = mean_squared_error(y_valid, predicted_valid_set)
                    print ("Mean Squared Error: ", result)
            
            #############################################################
            #                   OUTPUT
            #############################################################
            if mode==1:
                if use_validation_set:
                    df_filter_column[self.output_column] = float('nan')
                    df_filter_column.ix[train_indexes, self.output_column] = prediction
                    df_filter_column.ix[test_indexes, self.output_column] = predicted_test_set
                    df_filter_column[[self.output_column]].to_csv(workdir+self.output_filename)
                else:
                    df[self.output_column] = prediction
                    df[[self.output_column]].to_csv(workdir+self.output_filename)

                print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(original_row_count))
            else:
                print ("fitness="+str(weighted_result))               # main fitness metric
                print ("out_result_1="+str(weighted_auc))             # ROC AUC in train/test CV
                print ("out_result_2="+str(result))                   # main fitness on Validation
                print ("out_result_3="+str(result_roc_auc))           # ROC AUC on Validation

            
ev_agent_{id} = cls_ev_agent_{id}()

