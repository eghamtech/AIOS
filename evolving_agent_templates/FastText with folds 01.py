#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=13
#key=fields_to_use;  type=random_int;  from=13;  to=13;  step=1
#key=map_dict;  type=random_from_set;  set=True
#key=field_ev_prefix;  type=random_from_set;  set=ev_field_fasttext
#key=field_ev_prefix_use_source_names;  type=random_from_set;  set=True
#key=nfolds;  type=random_int;  from=3;  to=3;  step=1
#key=random_folds;  type=random_from_set;  set=True
#key=random_folds_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=use_validation_set;  type=random_from_set;  set=True
#key=random_valid;  type=random_from_set;  set=True
#key=random_valid_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=random_valid_folds;  type=random_int;  from=10;  to=10;  step=1
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
#key=include_columns_type;  type=random_from_set;  set=is_dict_only
#key=include_columns_containing;  type=random_from_set;  set=
#key=ignore_columns_containing;  type=random_from_set;  set=%ev_field%
#key=objective_multiclass;  type=random_from_set;  set='multiclass'
#key=objective_regression;  type=random_from_set;  set='regression_l1'
#key=loss_function;  type=random_from_set;  set='softmax','ns','hs'
#key=learning_rate;  type=random_float;  from=0.001;  to=1;  step=0.001
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
#key=clean_text;  type=random_int;  from=0;  to=1;  step=1
#key=use_float32_dtype; type=random_from_set;  set=True
#key=min_perf_criteria;  type=random_float;  from=0.6;  to=0.6;  step=0.1
#key=use_thresholds_train; type=random_from_set;  set=True
#key=print_to_html; type=random_from_set;  set=True
#key=print_tables; type=random_from_set;  set=False
#end_of_genes_definitions

# AICHOO OS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

# this agent concatenates given columns into sentences by rows and applies FastText to learn the target
# if column is a text one and map_dict==True it loads corresponding text from its dictionary
# if column is just numeric or map_dict==False then the number is converted into text

class cls_ev_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import math
    import os.path, bz2, pickle
    import dateutil
    import calendar
    from sklearn.externals import joblib

    # obtain a unique ID for the current instance
    result_id = {id}
    # obtain random field (same for all instances within the evolution) which will be the prediction target for this instance/evolution
    target_definition = "{field_to_predict}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    target_col  = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]

    # obtain random selection of fields; number of fields to be selected specified in data:length gene for this instance
    data_defs     = {data}
    fields_to_use = {fields_to_use}
    start_fold    = {start_fold}
    nfolds        = {nfolds}
    map_dict      = {map_dict}
    field_ev_prefix_use_source_names = {field_ev_prefix_use_source_names}
    
    dicts_agent   = {}         # various dictionary to be saved as part of model
    
    # if filter columns are specified then training and validation sets will be selected based on filter criteria
    # based on filter criteria training + validation sets will not necessarily constitute all data, the remainder will be called "test set"
    filter_column   = "{filter_column}"
    filter_column_2 = "{filter_column_2}"
    
    # fields matching the specified prefix will not be used in the model
    ignore_columns_containing  = "{ignore_columns_containing}"
    # include only fields matching string e.g., only properly scaled columns should be used with MLP
    include_columns_containing = "{include_columns_containing}"
    
    objective_multiclass = {objective_multiclass}
    objective_regression = {objective_regression}
    
    # defines whether to use Lperc/Uperc from train summary or validation summary
    use_thresholds_train = {use_thresholds_train}
    print_tables  = {print_tables}
    print_to_html = {print_to_html}
    
    use_validation_set = {use_validation_set}
    use_float32_dtype  = {use_float32_dtype}
    min_perf_criteria  = {min_perf_criteria}
   
    def __init__(self):
        from datetime import datetime
        import pyfasttext as ft
        # remove the target field for this instance from the data used for training
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)
        
        # create new field name based on "field_ev_prefix" with unique instance ID
        # and filename to save new field data
        self.field_ev_prefix = "{field_ev_prefix}"        
        if self.field_ev_prefix_use_source_names:                   
            # concatenate all source column names into new field prefix
            for i in range(0,self.fields_to_use):
                col_name = self.data_defs[i].split("|")[0]
                self.field_ev_prefix = self.field_ev_prefix + '_' + col_name
        
        self.output_column   = self.field_ev_prefix + '_' + str(self.result_id)
        self.output_filename = self.output_column + ".csv"
        
        # if saved models for the target field already exist then load them from filesystem              
        if self.os.path.isfile(workdir + self.output_column + '_dicts.model'):
            rfile = self.bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'r')
            self.dicts_agent = self.pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.output_column + ' dictionaries model loaded')
            
            self.predictors = []
            if self.dicts_agent['params']['random_folds'] == False:
                from_fold = self.start_fold
                to_fold   = self.nfolds
            else:
                from_fold = 0
                to_fold   = 3                                        # use fixed 3 saved models to make any prediction

            for fold in range(from_fold, to_fold):
                if self.os.path.isfile(workdir + self.output_column + "_fold" + str(fold) + ".model"):
                    # predictor_stored = ft.FastText(workdir + self.output_column + "_fold" + str(fold) + ".model.bin")
                    predictor_stored = self.joblib.load(workdir + self.output_column + "_fold" + str(fold) + ".model")
                    self.predictors.append(predictor_stored)
                    print (str(datetime.now()), self.output_column + ' fold ' + str(fold) + ' predictor model loaded')    
                      
        # obtain columns definitions to filter data set by
        if self.is_set(self.filter_column):
            self.filter_filename = self.filter_column.split("|")[1]
            self.filter_column   = self.filter_column.split("|")[0]
      
        if self.is_set(self.filter_column_2):
            self.filter_filename_2 = self.filter_column_2.split("|")[1]
            self.filter_column_2   = self.filter_column_2.split("|")[0]

    
    def is_set(self, s):
        return len(s)>0 and s!="0"

    def is_use_column(self, s):
        # AIOS Kernel now selects columns using agent parameters
        # so no need to filter inside the agent       
        if s.find(self.target_col)>=0:  # ignore columns that contain target_col as they are a derivative of the target
            return False 
        
        return True

    def timestamp(self, x):
        return self.calendar.timegm(self.dateutil.parser.parse(x).timetuple())

    def print_tbl(self, mesg):
        if self.print_tables:
            print (mesg)
    
    def print_html(self, df, max_rows=50, max_cols=25, jup_notebook=True):
        if self.print_to_html:
            print (df.to_html(max_rows=max_rows,max_cols=max_cols))
        elif jup_notebook:
            display (df)
        else:
            print (df)
            
    def my_log_loss(self, a, b):
        eps = 1e-9
        sum1 = 0.0
        for k in range(0, len(a)):
            bx = min(max(b[k],eps), 1-eps)
            sum1 += 1.0 * a[k] * self.math.log(bx) + 1.0 * (1 - a[k]) * self.math.log(1 - bx)
        return -sum1/len(a)
    
    def list_mean(self, lst, precision=4):
        return self.np.round(sum(lst)/float(len(lst)), decimals=precision)
    
    def clean_text_v1(self, string):
        import re
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " 's", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " ( ", string) 
        string = re.sub(r"\)", " ) ", string) 
        string = re.sub(r"\?", " ? ", string) 
        string = re.sub(r"\s{2,}", " ", string)       
        return  string.strip().lower()
       
    def ft_predict_proba(self, ft_predictor, xt, k, params):
        try:
            xt = xt.to_string(header=False, index=False, index_names=False).split('\n')
            # xt = [' '.join(element.split()) for element in xt]
            pred = ft_predictor.predict_proba(xt, k=k)
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
                    
    def load_columns(self, map_dict=True, clean_text=1):
        from datetime import datetime
        # start from loading the target field
        df_all = self.pd.read_csv(workdir+self.target_file, usecols=[self.target_col])[[self.target_col]]

        columns_new = [self.target_col]
        columns     = [self.target_col]
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        print (str(datetime.now()), " start loading data")
        cols_count     = 0
        block_progress = 0
        block          = int(self.fields_to_use/20)

        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if self.is_use_column(col_name):
                cols_count+=1
                if cols_count > self.fields_to_use:
                    break

                df_col = self.pd.read_csv(workdir+file_name, usecols=[col_name])[[col_name]]       # read column from csv file
                
                # if column has associated dictionary csv then it's a text column, replace column with actual text
                dict_file_name = workdir+'dict_'+col_name+'.csv'
                if self.os.path.isfile(dict_file_name) and map_dict:
                    dict1 = self.pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()  # load dictionary
                    df_col[col_name] = df_col[col_name].map(dict1)                                                         # map and replace
                                                   
                    if clean_text == 1:
                        df_col[col_name] = df_col[col_name].astype(str).apply(self.clean_text_v1)
                else:
                    if df_col[col_name].dtype == self.np.float64 and self.use_float32_dtype:           # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(self.np.float32)

                df_all = df_all.merge(df_col, left_index=True, right_index=True)                       # add column to the overall dataframe

                block_progress += 1
                if (block_progress >= block):
                    block_progress = 0
                    print (str(datetime.now()), " data loaded: ", round(cols_count/self.fields_to_use*100,0), "%")

                # some columns may appear multiple times in data_defs as inhereted from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                columns.append(col_name)
                ncol_count = columns.count(col_name)
                if ncol_count==1:
                    columns_new.append(col_name)
                else:
                    columns_new.append(col_name+"_v"+str(ncol_count))

        # rename columns in df to unique names
        df_all.columns = columns_new
        print (str(datetime.now()), " data loaded", len(df_all), "rows; ", len(df_all.columns), "columns")
        return df_all
        
        
    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        # df_add shouldn't contain columns with text values - only numeric
        # by this stage all text fields should have been converted to dictionary values by previous agents that created such fields in AIOS
        # since this agent works with text fields, actual text values will be loaded from dictionary files
        columns_new = []
        columns     = []
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        cols_count = 0
        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]
            
            if self.is_use_column(col_name):
                cols_count+=1
                if cols_count > self.fields_to_use:
                    break
                    
                # assemble dataframe column by column             
                df_col = df_add[[col_name]]
                
                # if column has associated dictionary csv then it's a text column, replace column with actual text
                dict_file_name = workdir+'dict_'+col_name+'.csv'
                if self.os.path.isfile(dict_file_name) and self.map_dict:
                    dict1 = self.pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()  # load dictionary
                    df_col[col_name] = df_col[col_name].map(dict1)                                                         # map and replace
                                   
                    if self.dicts_agent['params']['clean_text'] == 1:
                        df_col[col_name] = df_col[col_name].astype(str).apply(self.clean_text_v1)    
                else:
                    if df_col[col_name].dtype == self.np.float64 and self.use_float32_dtype:                                    # downcast to save memory if needed
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
        if self.dicts_agent['params']['objective'] == self.objective_multiclass:
            # create a list of lists depending on number of classes used for training 
            # as each prediction is a list of values against each class
            pred = [self.np.zeros(self.dicts_agent['params']['num_class']) for i in range(len(df))]
         
        # apply model from each fold created during training and sum their predictions
        for fold in range(0, len(self.predictors)):
            pred += self.ft_predict_proba( self.predictors[fold], df, k=self.dicts_agent['params']['num_class'], params=self.dicts_agent['params'] )
                    
        if self.dicts_agent['params']['objective'] == self.objective_multiclass:
            # select class with largest total value in case of multiclass
            pred = self.np.argmax(pred, axis=1)
        else:
            # average prediction over all folds in case of binary or regression   
            pred = pred / len(self.predictors)
        
        df_add[self.output_column] = pred
        

    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        # import fasttext
        import pyfasttext as ft
        from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, log_loss
        from sklearn.metrics import confusion_matrix, f1_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import StratifiedShuffleSplit
        from math import sqrt
        from datetime import datetime
        print ("enter run mode " + str(mode))  # 0=work for fitness only;  1=make new output field
        
        # prepare all parameters   
        # https://fasttext.cc/docs/en/options.html
        params                           = {}
        params['learning_rate']          = {learning_rate}    
        params['lr_update_rate']         = {lr_update_rate}
        params['size_of_word_vectors']   = {size_of_word_vectors}
        params['size_of_context_window'] = {size_of_context_window}
        params['epoch']                  = {epoch}
        params['negatives_sampled']      = {negatives_sampled}
        params['word_ngrams']            = {word_ngrams}            # max length of word ngram [1]
        params['loss_function']          = {loss_function}
        params['bucket']                 = {bucket}                 # number of buckets [2000000]
        params['cutoff']                 = {cutoff}
        params['dsub']                   = {dsub}
        params['min_char_ngrams']        = {min_char_ngrams}        # min length of char ngram [3]
        params['max_char_ngrams']        = {max_char_ngrams}        # max length of char ngram [6]
        params['min_word_occurences']    = {min_word_occurences}    # minimal number of word occurrences [5]
        params['min_count_label']        = {min_count_label}        # minimal number of label occurrences [0]
        params['qnorm']                  = {qnorm}
        params['qout']                   = {qout}
        params['verbose']                = 2
        params['num_threads']            = {num_threads}
        params['sampling_threshold']     = {sampling_threshold}
        params['clean_text']             = {clean_text}
        
        params['random_valid']           = {random_valid}
        params['random_valid_size']      = {random_valid_size}
        params['random_valid_folds']     = {random_valid_folds}
        params['random_folds']           = {random_folds}
        params['random_folds_size']      = {random_folds_size}
        
        # obtain indexes for train and remainder sets
        # load target column as it may be needed for filtering and removing NaN targets from training
        df_filter_column       = self.pd.read_csv(workdir+self.target_file, usecols=[self.target_col])
        filter_condition_train = df_filter_column[self.target_col].notnull()
        
        # applying specified filters
        if self.is_set(self.filter_column):
            # load columns to filter by
            df_t = self.pd.read_csv(workdir+self.filter_filename, usecols = [self.filter_column])
            df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)
            
            filter_condition_train = self.np.logical_and( filter_condition_train,
                                        self.np.logical_and( df_filter_column[self.filter_column]>={train_set_from}, 
                                                             df_filter_column[self.filter_column]<{train_set_to} ) )
            
            # two filter columns specified
            if self.is_set(self.filter_column_2):
                df_t = self.pd.read_csv(workdir+self.filter_filename_2, usecols = [self.filter_column_2])
                df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)
                
                condition2 = self.np.logical_and( df_filter_column[self.filter_column_2]>={train_set_from_2}, df_filter_column[self.filter_column_2]<{train_set_to_2} )
                filter_condition_train = self.np.logical_and( filter_condition_train, condition2 ) 
            
        train_filtered_indexes = df_filter_column[filter_condition_train].index.tolist()
        remainder_set_indexes  = df_filter_column[self.np.logical_not(filter_condition_train)].index.tolist()   # remainder which is not in train
        
        # load specified in data_defs colums of data up-to fields_to_use quantity
        df_all = self.load_columns(map_dict=self.map_dict, clean_text=params['clean_text'])
        original_row_count = len(df_all)
        
        # analyse target column whether it is binary which may result in different loss function used
        target_classes = df_all[df_all[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()
        is_binary = target_classes==[0, 1]
            
        if is_binary:
            print ("detected binary target: use AUC/LOGLOSS")
            params['objective'] = 'binary'
            params['num_class'] = 2                              # FT is multi-class predictor, hence even in binary it has 2 classes
            params['metric']    = ['auc', 'binary_logloss']
        elif self.is_set(self.objective_multiclass):
            print ("detected multi-class target: use Multi-LogLoss/Error; " + str(len(target_classes)) + " classes")
            params['objective'] = self.objective_multiclass
            params['num_class'] = max(target_classes) + 1        # requires all int numbers from 0 to max to be classes
            params['metric']    = ['multi_logloss','multi_error']
        else:
            print ("detected regression target: use RMSE/MAE")
            params['objective'] = self.objective_regression
            params['metric']    = ['rmse','mae']

        self.dicts_agent['params'] = params
            
        train_sets_ix                = []
        valid_sets_ix                = []
        predictors_all               = []
        weighted_result_folds        = []
        weighted_auc_folds           = []
        valid_result_folds           = []
        valid_result_auc_folds       = []    
        
        fold_all = 0
        # repeat cross-validation multiple times with different validation set each time
        # applies only in case when params['random_valid'] == True
        for valid_fold in range(0,params['random_valid_folds']):
            print ()
            print (str(datetime.now())," ----- VALID FOLD: ", valid_fold)
            # obtain indexes for validation set if required
            # applying specified filters
            valid_indexes = []
            if self.use_validation_set:
                # assemble condition for filtering validation set
                filter_condition_valid = df_filter_column[self.target_col].notnull()

                if self.is_set(self.filter_column):
                    filter_condition_valid = self.np.logical_and( filter_condition_valid,
                                                self.np.logical_and( df_filter_column[self.filter_column]>={valid_set_from}, 
                                                                     df_filter_column[self.filter_column]<{valid_set_to} ) )
                    # two filter columns specified
                    if self.is_set(self.filter_column_2):
                        condition2 = self.np.logical_and( df_filter_column[self.filter_column_2]>={valid_set_from_2}, df_filter_column[self.filter_column_2]<{valid_set_to_2} )
                        filter_condition_valid = self.np.logical_and( filter_condition_valid, condition2 )

                if params['random_valid'] == False:
                    # select validation based on fixed filter - may intersect with test or remainder set
                    train_sets_ix.append( train_filtered_indexes )
                    valid_sets_ix.append( df_filter_column[filter_condition_valid].index.tolist() )
                else:
                    # apply stratified random selection to previously filtered train set
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=params['random_valid_size'])
                    y  = df_filter_column[df_filter_column.index.isin(train_filtered_indexes)][[self.target_col]]
                    iy = y.reset_index(level=0)                                                     # create copy, save existing index in 'index' column and reset index 
                    y.reset_index(drop=True, inplace=True)                                          # reset index because StratifiedShuffleSplit will reset index anyway

                    for train_ix, valid_ix in sss.split(self.np.zeros(len(y)), y):
                        train_sets_ix.append( iy[iy.index.isin(train_ix)]['index'].tolist() )       # obtain original indexes from saved copy of labels with original indexes
                        valid_sets_ix.append( iy[iy.index.isin(valid_ix)]['index'].tolist() )       # can't use train_ix, valid_ix directly because they refer to new index reset during shuffling
                        print ('TRAIN target mean: ', df_filter_column[df_filter_column.index.isin(train_sets_ix[valid_fold])][self.target_col].mean().round(3))
                        print ('VALID target mean: ', df_filter_column[df_filter_column.index.isin(valid_sets_ix[valid_fold])][self.target_col].mean().round(3))

            # save indexes used for splits
            self.dicts_agent['train_sets_ix']    = train_sets_ix
            self.dicts_agent['remainder_set_ix'] = remainder_set_indexes
            self.dicts_agent['valid_sets_ix']    = valid_sets_ix

            print ("Length of train set:",          len(train_sets_ix[valid_fold]))
            print ("Length of test/remainder set:", len(remainder_set_indexes))
            print ("Length of validation set:",     len(valid_sets_ix[valid_fold]))


            # duplicate originally loaded data
            df = df_all.copy()
            # use previously calculated indexes to select train, validation and remainder sets
            df_test       = df[df.index.isin(remainder_set_indexes)]
            df_test.reset_index(drop=True, inplace=True)

            if self.use_validation_set:        
                df_valid  = df[df.index.isin(valid_sets_ix[valid_fold])]
                df_valid.reset_index(drop=True, inplace=True)
                # initialise prediction column for validation as it will be aggregate prediction from multiple folds
                predicted_valid_set = self.np.zeros(len(df_valid))                
                # Multi-class case: initialise prediction list of lists depending on number of classes 
                # as each prediction is a list of values against each class
                if params['objective'] == self.objective_multiclass:
                    predicted_valid_set = [self.np.zeros(params['num_class']) for i in range(len(df_valid))]

            df            = df[df.index.isin(train_sets_ix[valid_fold])]
            df.reset_index(drop=True, inplace=True)

            # initialise prediction column for main train set as it will be aggregate prediction from multiple folds
            prediction = self.np.zeros(len(df))
            # initialise prediction column for remainder set as it will be aggregate prediction from multiple folds   
            predicted_test_set  = self.np.zeros(len(df_test))
            # Multi-class case: initialise prediction list of lists depending on number of classes 
            # as each prediction is a list of values against each class
            if params['objective'] == self.objective_multiclass:
                prediction         = [self.np.zeros(params['num_class']) for i in range(len(df))]
                predicted_test_set = [self.np.zeros(params['num_class']) for i in range(len(df_test))]           

            #############################################################
            #                   MAIN LOOP
            #############################################################

            weighted_result = 0
            weighted_auc    = 0
            count_records_notnull = 0

            if params['random_folds'] == False:
                # divide training data into nfolds of size block
                block = int(len(df)/self.nfolds)
                # select folds sequentially in existing index order
                for fold in range(self.start_fold, self.nfolds):
                    print ()
                    print (str(datetime.now())," Train/Test FOLD: ", fold)
                    range_start = fold*block
                    range_end   = (fold+1)*block
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
                        self.os.remove(workdir + self.output_column + "_fold" + str(fold) + ".model.bin")
                        self.os.remove(workdir + self.output_column + "_fold" + str(fold) + ".model.vec")            
                        self.os.remove(workdir + self.output_column + '_train.tmp')
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
                            
                            if self.print_tables:
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)
                        elif self.is_set(self.objective_multiclass):
                            pred_classes = self.np.argmax(pred, axis=1)
                            result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                            result_acc_score  = accuracy_score(y_test, pred_classes)
                            result_cm = confusion_matrix(y_test, pred_classes)
                            result_cr = classification_report(y_test, pred_classes)
                            
                            if self.print_tables:
                                print ("Precision score: ", result_prec_score)
                                print ("Accuracy score: ",  result_acc_score)
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)
                            # result = predictor.best_score['valid_0']['multi_logloss']
                            result_roc_auc = f1_score(y_test, pred_classes, average='weighted')
                            result = result_roc_auc
                        else:
                            result = sum(abs(y_test-pred))/len(y_test)
                            #result = sqrt(mean_squared_error(y_test, pred))
                    except Exception as e:
                        print ('Evaluation error: ', e)

                    print ("result: ", result)

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
                    count_records_notnull += len(pred)

                    # predict all examples in the original test set which may include erroneous examples previously removed
                    pred = self.ft_predict_proba( x_test_orig.drop(self.target_col, axis=1), k=k, params=params )
                    print (str(datetime.now())," original test set predicted")

                    if params['objective'] == self.objective_multiclass:
                        prediction[range_start:range_end] = self.np.argmax(pred, axis=1)
                    else:
                        prediction[range_start:range_end] = pred

                    # predict validation and remainder sets examples
                    if self.use_validation_set:
                        pred = self.ft_predict_proba( df_valid.drop(self.target_col, axis=1), k=k, params=params )        
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
            else:
                # select folds using random shuffle and stratify
                sss = StratifiedShuffleSplit(n_splits=self.nfolds, test_size=params['random_folds_size'])
                y   = df[[self.target_col]]
              
                predictors = []
                for train_ix, test_ix in sss.split(self.np.zeros(len(y)), y):
                    fold_all += 1
                    print ()
                    print (str(datetime.now())," Train/Test FOLD: ", fold_all)

                    x_test  = df[df.index.isin(test_ix)]
                    x_test.reset_index(drop=True, inplace=True)
                    x_train = df[df.index.isin(train_ix)]

                    print ("x_test  rows count: " + str(len(x_test)))
                    print ("x_train rows count: " + str(len(x_train)))
                    
                    # convert target to multi-class labels
                    x_train[self.target_col + '_label'] = '__label__'
                    x_train[self.target_col] = x_train[self.target_col + '_label'] + x_train[self.target_col].fillna(-1).astype(int).astype(str)
                    x_train.drop(self.target_col + '_label', 1, inplace=True)
                    
                    y_test = x_test[self.target_col]                     # save test real test labels as needed later for evaluation
                    
                    # create text files as required by FastText
                    x_train.to_csv(workdir+self.output_column+'_train.tmp', sep=' ', header=False, index=False) 

                    try:
                        print (str(datetime.now())," start learning")
                        predictor = ft.FastText(label='__label__')
                        predictor.supervised(  input=workdir+self.output_column+'_train.tmp', output=workdir + self.output_column + "_fold" + str(fold_all) + ".model",
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
                        self.os.remove(workdir + self.output_column + "_fold" + str(fold_all) + ".model.bin")
                        self.os.remove(workdir + self.output_column + "_fold" + str(fold_all) + ".model.vec")            
                        self.os.remove(workdir + self.output_column + '_train.tmp')
                        return

                    # remove temp files created during learning
                    self.os.remove(workdir + self.output_column + '_train.tmp')
                    self.os.remove(workdir + self.output_column + "_fold" + str(fold_all) + ".model.bin")
                    self.os.remove(workdir + self.output_column + "_fold" + str(fold_all) + ".model.vec")
            
                    if is_binary:
                        k = 2
                    else:
                        k = len(predictor.labels)

                    pred = self.ft_predict_proba( predictor, x_test.drop(self.target_col, axis=1), k=k, params=params )
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
                            
                            if self.print_tables:
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)
                        elif self.is_set(self.objective_multiclass):
                            pred_classes = self.np.argmax(pred, axis=1)
                            result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                            result_acc_score  = accuracy_score(y_test, pred_classes)
                            result_cm = confusion_matrix(y_test, pred_classes)
                            result_cr = classification_report(y_test, pred_classes)
                            
                            if self.print_tables:
                                print ("Precision score: ", result_prec_score)
                                print ("Accuracy score: ",  result_acc_score)
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)
                            # result = predictor.best_score['valid_0']['multi_logloss']
                            result_roc_auc = f1_score(y_test, pred_classes, average='weighted')
                            result = result_roc_auc
                        else:
                            result = sum(abs(y_test-pred))/len(y_test)
                            #result = sqrt(mean_squared_error(y_test, pred))
                    except Exception as e:
                        print ('Evaluation error: ', e)

                    print ("result: ", result)

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
                    count_records_notnull += len(pred)
                    
                    if result_roc_auc < self.min_perf_criteria:
                        print ("Minimum performance criteria: " + str(self.min_perf_criteria) + " not met! result_roc_auc: " + str(result_roc_auc))
                        return

                    predictors.append(    [predictor,result,result_roc_auc])
                    predictors_all.append([predictor,result,result_roc_auc])    # add predictors to global list across all validation folds
          
        
                predictors = self.pd.DataFrame(predictors, columns=['predictor','result','result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
                print ('\nFolds Performance Overall:')
                self.print_html( predictors, max_rows=50, max_cols=5 )

                best_predictor_idx  = predictors['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors['result_roc_auc'].idxmin()
                avg_predictor_idx   = self.np.argwhere(predictors['result_roc_auc']>=predictors['result_roc_auc'].mean())[0][0]
                predictors = [predictors['predictor'][worst_predictor_idx], predictors['predictor'][avg_predictor_idx], predictors['predictor'][best_predictor_idx]]

                x_test = df.drop(self.target_col, axis=1)
                    
                for fold in range(0, len(predictors)):
                    # predict entire train set
                    prediction += self.ft_predict_proba( predictors[fold], x_test, k=k, params=params )
                    # predict remainder set
                    if len(df_test) > 0:
                        predicted_test_set  += self.ft_predict_proba( predictors[fold], df_test.drop(self.target_col, axis=1), k=k, params=params )

                    # predict validation set
                    if self.use_validation_set:
                        predicted_valid_set += self.ft_predict_proba( predictors[fold], df_valid.drop(self.target_col, axis=1), k=k, params=params )

                prediction = prediction / len(predictors)
                predicted_test_set  = predicted_test_set  / len(predictors)
                predicted_valid_set = predicted_valid_set / len(predictors)
                
                
            weighted_result = weighted_result/count_records_notnull
            weighted_auc    = weighted_auc/count_records_notnull

            weighted_result_folds.append(weighted_result)
            weighted_auc_folds.append(weighted_auc)

            print ("\nweighted_result:", weighted_result)
            print ("weighted_auc:",      weighted_auc)

            # if multiclass convert list of lists into list of predicted labels
            if params['objective'] == self.objective_multiclass:             
                predicted_valid_set = self.np.argmax(predicted_valid_set, axis=1)
                predicted_test_set  = self.np.argmax(predicted_test_set, axis=1)
        
            if use_validation_set:
                print()
                print ("*************  VALIDATION SET RESULTS  *****************")
                print ("Length of validation set:", len(predicted_valid_set))

                # validation set may have missing labels (NAN), for metrics calc find subset with proper labels
                df_valid['predicted_valid_set'] = predicted_valid_set
                df_valid = df_valid[df_valid[self.target_col].notnull()]
                df_valid.reset_index(drop=True, inplace=True)
                y_valid             = df_valid[self.target_col].tolist()
                predicted_valid_set = df_valid['predicted_valid_set'].tolist()

                if is_binary:                        
                    try:
                        result = self.my_log_loss(y_valid, predicted_valid_set)
                        print ("LOGLOSS: ", result)
                        result_roc_auc = roc_auc_score(y_valid, predicted_valid_set)
                        print ("ROC AUC score: ", result_roc_auc)
                        
                        if self.print_tables:
                            result_cm = confusion_matrix(y_valid, (self.np.asarray(predicted_valid_set)>0.5))  # assume 0.5 probability threshold
                            print ("Confusion Matrix:\n",      result_cm)
                            result_cr = classification_report(y_valid, (self.np.asarray(predicted_valid_set)>0.5))
                            print ("Classification Report:\n", result_cr)
                    except Exception as e:
                        print (e)
                        return             # no point to carry on with more folds
                elif params['objective'] == self.objective_multiclass:
                    try:
                        result_prec_score = precision_score(y_valid, predicted_valid_set, average='weighted')
                        result_acc_score  = accuracy_score(y_valid, predicted_valid_set)
                        result_cm = confusion_matrix(y_valid, predicted_valid_set)
                        result_cr = classification_report(y_valid, predicted_valid_set)
                        
                        if self.print_tables:
                            print ("Precision score: ", result_prec_score)
                            print ("Accuracy score: ",  result_acc_score)
                            print ("Confusion Matrix:\n",      result_cm)
                            print ("Classification Report:\n", result_cr)
                            
                        result = 1 - result_prec_score
                        result_roc_auc = f1_score(y_valid, predicted_valid_set, average='weighted')
                    except Exception as e:
                        print (e)
                        return             # no point to carry on with more folds
                else:
                    #result = sum(abs(y_valid-predicted_valid_set))/len(y_valid)
                    #print ("MAE: ", result)
                    result = sqrt(mean_squared_error(y_valid, predicted_valid_set))
                    result_roc_auc = 1/result
                    print ("Root Mean Squared Error: ", result)
                    
                valid_result_folds.append(result)
                valid_result_auc_folds.append(result_roc_auc)         
                print ("\n************* END of VALIDATION SET RESULTS  ****************\n")
        
        print ('\nTrain/Valid Folds Predictor Performance Overall:')
        predictors_all = self.pd.DataFrame(predictors_all, columns=['predictor','result','result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
        self.print_html( predictors_all, max_rows=50, max_cols=5 )
        
        #############################################################
        #                   OUTPUT
        #############################################################

        if mode==1:
            # save dictionary of all auxiliry data and params into file
            sfile = self.bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'w')
            self.pickle.dump(self.dicts_agent, sfile) 
            sfile.close()
            
            if params['random_folds'] == False:          
                df_filter_column[self.output_column] = float('nan')
                df_filter_column.loc[train_sets_ix[valid_fold], self.output_column] = prediction
                df_filter_column.loc[remainder_set_indexes,     self.output_column] = predicted_test_set

                if self.use_validation_set:
                    df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column] = predicted_valid_set
            else:
                # select 3 models from all train/test/valid folds
                best_predictor_idx  = predictors_all['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors_all['result_roc_auc'].idxmin()
                avg_predictor_idx   = self.np.argwhere(predictors_all['result_roc_auc']>=predictors_all['result_roc_auc'].mean())[0][0]
                predictors = [predictors_all['predictor'][worst_predictor_idx], predictors_all['predictor'][avg_predictor_idx], predictors_all['predictor'][best_predictor_idx]]
                
                x_test = df_all.drop(self.target_col, axis=1)
                prediction = self.np.zeros(len(x_test))
                if params['objective'] == self.objective_multiclass:
                    prediction = [self.np.zeros(params['num_class']) for i in range(len(x_test))]
                
                for fold in range(0, len(predictors)):
                    # predict entire data set
                    prediction += self.ft_predict_proba( predictors[fold], x_test, k=k, params=params )
                    self.joblib.dump(predictors[fold], workdir + self.output_column + "_fold" + str(fold) + ".model")

                # if multiclass convert list of lists into list of predicted labels
                if params['objective'] == self.objective_multiclass:             
                    df_filter_column[self.output_column] = self.np.argmax(prediction, axis=1)
                else:
                    df_filter_column[self.output_column] = prediction / len(predictors)
                    
            df_filter_column[[self.output_column]].to_csv(workdir+self.output_filename)
            print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(original_row_count))
            
            print ("b_fitness="   +str(1-self.list_mean(weighted_auc_folds)*self.list_mean(valid_result_auc_folds)))
            print ("b_result_1="+str(self.list_mean(weighted_result_folds)))
            print ("b_result_2="+str(self.list_mean(weighted_auc_folds)))
            print ("b_result_3="+str(self.list_mean(valid_result_folds)))
            print ("b_result_4="+str(self.list_mean(valid_result_auc_folds)))
        else:
            print ("fitness="     +str(1-self.list_mean(weighted_auc_folds)*self.list_mean(valid_result_auc_folds)))  # main fitness metric
            print ("out_result_1="+str(self.list_mean(weighted_result_folds)))                                        # Log Loss in train/test CV
            print ("out_result_2="+str(self.list_mean(weighted_auc_folds)))                                           # ROC AUC in train/test CV
            print ("out_result_3="+str(self.list_mean(valid_result_folds)))                                           # main fitness on Validation
            print ("out_result_4="+str(self.list_mean(valid_result_auc_folds)))                                       # ROC AUC on Validation

                      
ev_agent_{id} = cls_ev_agent_{id}()
