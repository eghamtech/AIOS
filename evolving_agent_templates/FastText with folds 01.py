#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=13
#key=fields_to_use;  type=random_int;  from=13;  to=13;  step=1
#key=nfolds;  type=random_int;  from=10;  to=10;  step=1
#key=use_validation_set;  type=random_from_set;  set=False
#key=filter_column;  type=random_from_set;  set=field|field.csv
#key=train_set_from;  type=random_from_set;  set=self.timestamp('2013-11-01')
#key=train_set_to;  type=random_from_set;  set=self.timestamp('2014-11-01')
#key=valid_set_from;  type=random_from_set;  set=self.timestamp('2014-11-01')
#key=valid_set_to;  type=random_from_set;  set=self.timestamp('2016-11-01')
#key=filter_column_2;  type=random_from_set;  set=
#key=train_set_from_2;  type=random_from_set;  set=
#key=train_set_to_2;  type=random_from_set;  set=
#key=valid_set_from_2;  type=random_from_set;  set=
#key=valid_set_to_2;  type=random_from_set;  set=
#key=ignore_columns_containing;  type=random_from_set;  set=%ev_field%
#key=include_columns_containing;  type=random_from_set;  set=
#key=objective_multiclass;  type=random_from_set;  set='multiclass'
#key=objective_regression;  type=random_from_set;  set='regression_l1'
#key=loss_function;  type=random_from_set;  set='softmax','ns','hs'
#key=learning_rate;  type=random_float;  from=0.001;  to=0.5;  step=0.001
#key=sampling_threshold;  type=random_float;  from=0.00005;  to=0.0002;  step=0.00001
#key=lr_update_rate;  type=random_int;  from=10;  to=300;  step=1
#key=size_of_word_vectors;  type=random_int;  from=5;  to=150;  step=1
#key=size_of_context_window;  type=random_int;  from=1;  to=20;  step=1
#key=epoch;  type=random_int;  from=3;  to=25;  step=1
#key=negatives_sampled;  type=random_int;  from=1;  to=20;  step=1
#key=word_ngrams;  type=random_int;  from=1;  to=5;  step=1
#key=bucket;  type=random_int;  from=500000;  to=4000000;  step=1000
#key=cutoff;  type=random_int;  from=0;  to=0;  step=1
#key=dsub;  type=random_int;  from=1;  to=4;  step=1
#key=min_char_ngrams;  type=random_int;  from=0;  to=4;  step=1
#key=max_char_ngrams;  type=random_int;  from=0;  to=8;  step=1
#key=min_word_occurences;  type=random_int;  from=2;  to=10;  step=1
#key=min_count_label;  type=random_int;  from=0;  to=0;  step=1
#key=qnorm; type=random_from_set;  set=False
#key=qout; type=random_from_set;  set=False
#key=start_fold;  type=random_from_set;  set=0
#key=num_threads;  type=random_int;  from=1;  to=1;  step=1
#key=use_float32_dtype; type=random_from_set;  set=True
#end_of_genes_definitions

# AICHOO OS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

# this agent concatenates given columns into sentences by rows and applies FastText to learn the target
# if column is a text one it loads corresponding text from its dictionary
# if column is just numeric then the number is converted into text

class cls_ev_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import math
    import os.path, bz2, pickle
    import dateutil
    import calendar

    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "field_ev_prefix" with unique instance ID
    # and filename to save new field data
    field_ev_prefix = "ev_field_fasttext_"
    output_column = field_ev_prefix + str(result_id)
    output_filename = output_column + ".csv"

    # obtain random field (same for all instances within the evolution) which will be the prediction target for this instance/evolution
    target_definition = "{field_to_predict}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    target_col = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]

    # obtain random selection of fields; number of fields to be selected specified in data:length gene for this instance
    data_defs = {data}
    fields_to_use = {fields_to_use}
    start_fold = {start_fold}
    nfolds = {nfolds}
    
    # if filter columns are specified then training and validation sets will be selected based on filter criteria
    # based on filter criteria training + validation sets will not necessarily constitute all data, the remainder will be called "test set"
    filter_column = "{filter_column}"
    filter_column_2 = "{filter_column_2}"
    
    # fields matching the specified prefix will not be used in the model
    ignore_columns_containing = "{ignore_columns_containing}"
    # include only fields matching string e.g., only properly scaled columns should be used with MLP
    include_columns_containing = "{include_columns_containing}"
    
    objective_multiclass = {objective_multiclass}
    objective_regression = {objective_regression}
   
    def __init__(self):
        from datetime import datetime
        import pyfasttext as ft
        # remove the target field for this instance from the data used for training
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)
        
        # if saved models for the target field already exist then load them from filesystem
        self.predictors = []
        for fold in range(self.start_fold, self.nfolds):
            if self.os.path.isfile(workdir + self.output_column + "_fold" + str(fold) + ".model.bin"):
                predictor_stored = ft.FastText(workdir + self.output_column + "_fold" + str(fold) + ".model.bin")
                self.predictors.append(predictor_stored)
                print (str(datetime.now()), self.output_column + ' fold ' + str(fold) + ' predictor model loaded')
  
        if self.os.path.isfile(workdir + self.output_column + '_params.model'):
            rfile = self.bz2.BZ2File(workdir + self.output_column + '_params.model', 'r')
            self.lgbm_params = self.pickle.load(rfile)
            rfile.close()    
            print (str(datetime.now()), self.output_column + ' model parameters loaded')
            
        # obtain columns definitions to filter data set by
        if self.is_set(self.filter_column):
            self.filter_filename = self.filter_column.split("|")[1]
            self.filter_column = self.filter_column.split("|")[0]
      
        if self.is_set(self.filter_column_2):
            self.filter_filename_2 = self.filter_column_2.split("|")[1]
            self.filter_column_2 = self.filter_column_2.split("|")[0]

    
    def is_set(self, s):
        return len(s)>0 and s!="0"

    def is_use_column(self, s):
        # determine whether given column should be ignored
        s = s.replace('%','')           # remove % used for pattern matching as now required to filter column by AIOS itself
        
        if s.find(self.target_col)>=0:  # ignore columns that contain target_col as they are a derivative of the target
            return False 
        # ignore other columns containing specified ignore parameter value
        if self.is_set(self.ignore_columns_containing) and s.find(self.ignore_columns_containing.replace('%',''))>=0:
            return False
        # include all columns if include parameter not specified
        if not self.is_set(self.include_columns_containing):
            return True
        # include columns specified in parameter
        if self.is_set(self.include_columns_containing) and s.find(self.include_columns_containing.replace('%',''))>=0:
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
    
    def ft_predict_proba(self, xt, k, params):
        try:
            xt = xt.to_string(header=False, index=False, index_names=False).split('\n')
            # xt = [' '.join(element.split()) for element in xt]
            pred = self.predictor.predict_proba(xt, k=k)
            # convert to list of dictionaries for each predicted item as predictor produces list of lists of tuples
            # [[('0', 0.712891), ('1', 0.285156)],
            #  [('0', 0.644531), ('1', 0.353516)],
            #  [('1', 0.945313), ('0', 0.0527344)],
            pred = [dict(r) for r in pred]   
            # [{'0': 0.712891, '1': 0.285156},
            #  {'0': 0.644531, '1': 0.353516},
            #  {'0': 0.0527344, '1': 0.945313},   

            if params['objective'] == self.objective_multiclass:
                pred = [list(r.values()) for r in pred]           # convert to list of lists with all labels probabilities
            else:
                pred = [r.get('1', 0) for r in pred]              # get probability for label '1' from each prediction item
        except Exception as e:
            print ('FastText Predict Proba error: ', e)
            pred = 0
        
        return pred
                    

    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        # df_add shouldn't contain columns with text values - only numeric
        # by this stage all text fields should have been converted to dictionary values by previous agents that created such fields in AIOS
        # since this agent works with text fields, actual text values will be loaded from dictionary files
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
                df_col = df_add[[col_name]]
                
                # if column has associated dictionary csv then it's a text column, replace column with actual text
                dict_file_name = workdir+'dict_'+col_name+'.csv'
                if self.os.path.isfile(dict_file_name):
                    dict1 = self.pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()  # load dictionary
                    df_col[col_name] = df_col[col_name].map(dict1)                                                         # map and replace
                else:
                    if df_col[col_name].dtype == self.np.float64 and use_float32_dtype:                                    # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(self.np.float32)
                        
                if cols_count==1:
                    df = df_col[[col_name]]
                else:
                    df = df.merge(df_col[[col_name]], left_index=True, right_index=True)
                
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
        
        # predict new data set in df applying model for each fold used for training
        pred = self.np.zeros(len(df))
        if self.lgbm_params['objective'] == self.objective_multiclass:
            # create a list of lists depending on number of classes used for training 
            # as each prediction is a list of values against each class
            pred = [self.np.zeros(self.lgbm_params['num_class']) for i in range(len(df))]
         
        # convert data set to list of strings as required by FT
        df = df.to_string(header=False, index=False, index_names=False).split('\n')
        # apply model from each fold created during training and sum their predictions
        for fold in range(self.start_fold, self.nfolds):
            pred_f = self.predictors[fold-self.start_fold].predict_proba(df, k=self.lgbm_params['num_class'])
            pred_f = [dict(r) for r in pred_f]                                        # convert to list of dictionaries for each predicted item as predictor produces list of lists of tuples
            if params['objective'] == self.objective_multiclass:
                pred_f = [list(r.values()) for r in pred_f]                           # convert to list of lists with all labels probabilities
            else:
                pred_f = [r.get('1', 0) for r in pred_f]                              # get probability for label '1' from each prediction item
            
            pred += pred_f                                                            # add to overall sum of all predictors output
        
        if self.lgbm_params['objective'] == self.objective_multiclass:
            # select class with largest total value in case of multiclass
            pred = self.np.argmax(pred, axis=1)
        else:
            # average prediction over all folds in case of binary or regression   
            pred = pred / (self.nfolds - self.start_fold)
        
        df_add[self.output_column] = pred
        

    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        # import fasttext
        import pyfasttext as ft
        from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, log_loss
        from sklearn.metrics import confusion_matrix, f1_score
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
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>=0, df_filter_column[self.filter_column]<360000)
                train_indexes = df_filter_column[condition1].index
                test_indexes = df_filter_column[self.np.logical_not(condition1)].index
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>=360000, df_filter_column[self.filter_column]<404290)
                validation_set_indexes = df_filter_column[condition1].index
            else:
                # two filter columns specified
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>=0, df_filter_column[self.filter_column]<360000)
                condition2 = self.np.logical_and(df_filter_column[self.filter_column_2]>=0, df_filter_column[self.filter_column_2]<0)
                train_indexes = df_filter_column[self.np.logical_and(condition1, condition2)].index
                test_indexes = df_filter_column[self.np.logical_not(self.np.logical_and(condition1, condition2))].index
                condition1 = self.np.logical_and(df_filter_column[self.filter_column]>=360000, df_filter_column[self.filter_column]<404290)
                condition2 = self.np.logical_and(df_filter_column[self.filter_column_2]>=0, df_filter_column[self.filter_column_2]<0)
                validation_set_indexes = df_filter_column[self.np.logical_and(condition1, condition2)].index
            
            print ("Length of train set:", len(train_indexes))
            print ("Length of test/remainder set:", len(test_indexes))
            print ("Length of validation set:", len(validation_set_indexes))
        
        # start from loading the target field
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
                
                # if column has associated dictionary csv then it's a text column, replace column with actual text
                dict_file_name = workdir+'dict_'+col_name+'.csv'
                if self.os.path.isfile(dict_file_name):
                    dict1 = self.pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()  # load dictionary
                    df_col[col_name] = df_col[col_name].map(dict1)                                                         # map and replace
                else:
                    if df_col[col_name].dtype == self.np.float64 and use_float32_dtype:                                    # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(self.np.float32)

                df = df.merge(df_col, left_index=True, right_index=True)                                                   # add column to the overall dataframe
                
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

        # rename columns in df to unique names
        df.columns = columns_new
        print (str(datetime.now()), " data loaded", len(df), "rows; ", len(df.columns), "columns")
        original_row_count = len(df)
        
        # analyse target column whether it is binary which may result in different loss function used
        target_classes = df[df[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()
        is_binary = target_classes==[0, 1]
            
        # prepare FT parameters   
        # https://fasttext.cc/docs/en/options.html
        params = {}
        params['learning_rate'] = {learning_rate}    
        params['lr_update_rate'] = {lr_update_rate}
        params['size_of_word_vectors'] = {size_of_word_vectors}
        params['size_of_context_window'] = {size_of_context_window}
        params['epoch'] = {epoch}
        params['negatives_sampled'] = {negatives_sampled}
        params['word_ngrams'] = {word_ngrams}                       # max length of word ngram [1]
        params['loss_function'] = {loss_function}
        params['bucket'] = {bucket}                                 # number of buckets [2000000]
        params['cutoff'] = {cutoff}
        params['dsub'] = {dsub}
        params['min_char_ngrams'] = {min_char_ngrams}               # min length of char ngram [3]
        params['max_char_ngrams'] = {max_char_ngrams}               # max length of char ngram [6]
        params['min_word_occurences'] = {min_word_occurences}       # minimal number of word occurrences [5]
        params['min_count_label'] = {min_count_label}               # minimal number of label occurrences [0]
        params['qnorm'] = {qnorm}
        params['qout'] = {qout}
        params['verbose'] = 2
        params['num_threads'] = {num_threads}
        params['sampling_threshold'] = {sampling_threshold}

        if is_binary:
            print ("detected binary target: use AUC/LOGLOSS")
            params['objective'] = 'binary'
            params['num_class'] = 2                              # FT is multi-class predictor, hence even in binary it has 2 classes
            params['metric'] = ['auc', 'binary_logloss']
        elif self.is_set(self.objective_multiclass):
            print ("detected multi-class target: use Multi-LogLoss/Error; " + str(len(target_classes)) + " classes")
            params['objective'] = self.objective_multiclass
            params['num_class'] = max(target_classes) + 1        # requires all int numbers from 0 to max to be classes
            params['metric'] = ['multi_logloss','multi_error']
        else:
            print ("detected regression target: use RMSE/MAE")
            params['objective'] = self.objective_regression
            params['metric'] = ['rmse','mae']

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
            # Multi-class case: initialise prediction list of lists depending on number of classes 
            # as each prediction is a list of values against each class
            if params['objective'] == self.objective_multiclass:
                predicted_valid_set = [self.np.zeros(params['num_class']) for i in range(len(self.df_valid))]
                predicted_test_set =  [self.np.zeros(params['num_class']) for i in range(len(df_test))]
                
        #############################################################
        #
        #                   MAIN LOOP
        #
        #############################################################

        # divide training data into nfolds of size block
        block = int(len(df)/self.nfolds)

        prediction = self.np.zeros(len(df))

        weighted_result = 0
        weighted_auc = 0
        count_records_notnull = 0

        for fold in range(self.start_fold, self.nfolds):
            print ()
            print (str(datetime.now())," FOLD", fold)
            range_start = fold*block
            range_end = (fold+1)*block
            if fold==self.nfolds-1:
                range_end = len(df)
            range_predict = range(range_start, range_end)
            print ("range start", range_start, "; range end ", range_end)

            x_test = df[df.index.isin(range_predict)]
            x_test.reset_index(drop=True, inplace=True)
            x_test_orig = x_test.copy()                           # save original test set before removing null values
            x_test = x_test[x_test[self.target_col].notnull()]    # remove examples that have no proper target label
            x_test.reset_index(drop=True, inplace=True)

            x_train = df[df.index.isin(range_predict)==False]
            x_train.reset_index(drop=True, inplace=True)
            x_train= x_train[x_train[self.target_col].notnull()]  # remove examples that have no proper target label
            x_train.reset_index(drop=True, inplace=True)

            print ("x_test rows count: " + str(len(x_test)))
            print ("x_train rows count: " + str(len(x_train)))

            # convert target to multi-class labels
            x_train[self.target_col + '_label'] = '__label__'
            x_train[self.target_col] = x_train[self.target_col + '_label'] + x_train[self.target_col].fillna(-1).astype(int).astype(str)
            x_train.drop(self.target_col + '_label', 1, inplace=True)
                   
            y_test = x_test[self.target_col]                     # save test real test labels as needed later for evaluation
            # x_test[self.target_col + '_label'] = '__label__'
            # x_test[self.target_col] = x_test[self.target_col + '_label'] + x_test[self.target_col].fillna(-1).astype(int).astype(str)
            # x_test.drop(self.target_col + '_label', 1, inplace=True)
        
            # create text files as required by FastText
            x_train.to_csv(workdir+self.output_column+'_train.tmp', sep=' ', header=False, index=False) 
            # x_test.to_csv(workdir+self.output_column+'_test.tmp', sep=' ', header=False, index=False)
            
            try:
                print (str(datetime.now())," start learning")
                self.predictor = ft.FastText(label='__label__')
                self.predictor.supervised(  input=workdir+self.output_column+'_train.tmp', output=workdir + self.output_column + "_fold" + str(fold) + ".model",
                                            bucket       = params['bucket'],
                                            cutoff       = params['cutoff'],
                                            dim          = params['size_of_word_vectors'],
                                            dsub         = params['dsub'],
                                            epoch        = params['epoch'],
                                            loss         = params['loss_function'],
                                            lr           = params['learning_rate'],
                                            lrUpdateRate = params['lr_update_rate'],
                                            maxn         = params['max_char_ngrams'],
                                            minCount     = params['min_word_occurences'],
                                            minCountLabel= params['min_count_label'],
                                            minn         = params['min_char_ngrams'],
                                            neg          = params['negatives_sampled'],
    #                                         qnorm        = params['qnorm'],
    #                                         qout         = params['qout'],
                                            t            = params['sampling_threshold'],
                                            thread       = params['num_threads'], 
                                            verbose      = params['verbose'],
                                            wordNgrams   = params['word_ngrams'],
                                            ws           = params['size_of_context_window']
                                         )
                print (str(datetime.now())," end learning")
            except Exception as e:
                print ('FastText Supervised error: ', e)
                print ("fitness="     +str(99999))
                return

            if is_binary:
                k = 2
            else:
                k = len(self.predictor.labels)
                
            pred = self.ft_predict_proba( x_test.drop(self.target_col, axis=1), k=k, params=params )
            print (str(datetime.now())," test set predicted")
            
            try:
                if is_binary:
                    result = self.my_log_loss(y_test, pred)
                    # show various metrics as per
                    # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                    result_roc_auc = roc_auc_score(y_test, pred)
                    result_cm = confusion_matrix(y_test, (self.np.asarray(pred)>0.5))  # assume 0.5 probability threshold
                    result_cr = classification_report(y_test, (self.np.asarray(pred)>0.5))
                    print ("ROC AUC score: ", result_roc_auc)
                    print ("Confusion Matrix:\n", result_cm)
                    print ("Classification Report:\n", result_cr)
                elif self.is_set(self.objective_multiclass):
                    pred_classes = self.np.argmax(pred, axis=1)
                    result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                    result_acc_score = accuracy_score(y_test, pred_classes)
                    result_cm = confusion_matrix(y_test, pred_classes)
                    result_cr = classification_report(y_test, pred_classes)
                    print ("Precision score: ", result_prec_score)
                    print ("Accuracy score: ", result_acc_score)
                    print ("Confusion Matrix:\n", result_cm)
                    print ("Classification Report:\n", result_cr)
                    # result = predictor.best_score['valid_0']['multi_logloss']
                    result_roc_auc = f1_score(y_test, pred_classes, average='weighted')
                else:
                    result = sum(abs(y_test-pred))/len(y_test)
                    #result = sqrt(mean_squared_error(y_test, pred))
            except Exception as e:
                print ('Evaluation error: ', e)
            
            print ("result: ", result)
            
            weighted_result += result * len(pred)
            weighted_auc += result_roc_auc * len(pred)
            count_records_notnull += len(pred)
        
            # predict all examples in the original test set which may include erroneous examples previously removed
            pred = self.ft_predict_proba( x_test_orig.drop(self.target_col, axis=1), k=k, params=params )
            print (str(datetime.now())," original test set predicted")
 
            if params['objective'] == self.objective_multiclass:
                prediction[range_start:range_end] = self.np.argmax(pred, axis=1)
            else:
                prediction[range_start:range_end] = pred
    
            # predict validation and remainder sets examples
            if use_validation_set:
                pred = self.ft_predict_proba( self.df_valid.drop(self.target_col, axis=1), k=k, params=params )        
                predicted_valid_set += pred
                print (str(datetime.now())," validation set predicted")
                
                pred = self.ft_predict_proba( df_test.drop(self.target_col, axis=1), k=k, params=params )          
                predicted_test_set += pred
                print (str(datetime.now())," remainder set predicted")
                
            if mode==0:
                self.os.remove(workdir + self.output_column + "_fold" + str(fold) + ".model.bin")
                self.os.remove(workdir + self.output_column + "_fold" + str(fold) + ".model.vec")
            
            self.os.remove(workdir + self.output_column + '_train.tmp')
            # self.os.remove(workdir + self.output_column + '_test.tmp')


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
            
            # if multiclass convert list of lists into list of predicted labels
            if params['objective'] == self.objective_multiclass:             
                predicted_valid_set = self.np.argmax(predicted_valid_set, axis=1)
                predicted_test_set = self.np.argmax(predicted_test_set, axis=1)
            
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
            elif params['objective'] == self.objective_multiclass:
                try:
                    result_prec_score = precision_score(y_valid, predicted_valid_set, average='weighted')
                    result_acc_score = accuracy_score(y_valid, predicted_valid_set)
                    result_cm = confusion_matrix(y_valid, predicted_valid_set)
                    result_cr = classification_report(y_valid, predicted_valid_set)
                    print ("Precision score: ", result_prec_score)
                    print ("Accuracy score: ", result_acc_score)
                    print ("Confusion Matrix:\n", result_cm)
                    print ("Classification Report:\n", result_cr)
                    result = 1 - result_prec_score
                    result_roc_auc = f1_score(y_valid, predicted_valid_set, average='weighted')
                except Exception as e:
                    print (e)
            else:
                #result = sum(abs(y_valid-predicted_valid_set))/len(y_valid)
                #print ("MAE: ", result)
                result = sqrt(mean_squared_error(y_valid, predicted_valid_set))
                print ("Root Mean Squared Error: ", result)
                
        #############################################################
        #                   OUTPUT
        #############################################################

        if mode==1:
            # save parameters used for training lGBM as needed during applying on new data
            sfile = self.bz2.BZ2File(workdir + self.output_column + '_params.model', 'w')
            self.pickle.dump(params, sfile) 
            sfile.close()
                        
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
            print ("fitness="     +str(self.np.round(1-weighted_auc, decimals = 4)))           # main fitness metric
            print ("out_result_1="+str(self.np.round(weighted_result, decimals = 4)))          # Log Loss in train/test CV
            print ("out_result_2="+str(self.np.round(weighted_auc, decimals = 4)))             # ROC AUC in train/test CV
            print ("out_result_3="+str(self.np.round(result, decimals = 4)))                   # main fitness on Validation
            print ("out_result_4="+str(self.np.round(result_roc_auc, decimals = 4)))           # ROC AUC on Validation
            
          
ev_agent_{id} = cls_ev_agent_{id}()
