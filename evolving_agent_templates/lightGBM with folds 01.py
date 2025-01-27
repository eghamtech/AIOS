#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=13
#key=fields_to_use;  type=random_int;  from=13;  to=13;  step=1
#key=field_ev_prefix;  type=random_from_set;  set=ev_field_lgbm_
#key=field_ev_prefix_use_target_name;  type=random_from_set;  set=True
#key=field_ev_prefix_use_source_names;  type=random_from_set;  set=True
#key=nfolds;  type=random_int;  from=3;  to=3;  step=1
#key=random_folds;  type=random_from_set;  set=True
#key=random_folds_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=use_validation_set;  type=random_from_set;  set=True
#key=random_valid;  type=random_from_set;  set=True
#key=random_valid_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=random_valid_folds;  type=random_int;  from=3;  to=3;  step=1
#key=models_to_save;  type=random_int;  from=-1;  to=-1;  step=1
#key=models_apply_on_all_data;  type=random_from_set;  set=True
#key=random_seed_init;  type=random_int;  from=1;  to=10000000;  step=1
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
#key=include_columns_type;  type=random_from_set;  set=
#key=include_columns_containing;  type=random_from_set;  set=
#key=ignore_columns_containing;  type=random_from_set;  set=%ev_field%
#key=objective_multiclass;  type=random_from_set;  set='multiclass','multiclassova'
#key=objective_regression;  type=random_from_set;  set='regression_l1','regression_l2','huber','fair','poisson','quantile','mape','gamma','tweedie'
#key=boosting_type;  type=random_from_set;  set='gbdt','rf','dart'
#key=learning_rate;  type=random_float;  from=0.001;  to=0.06;  step=0.001
#key=sub_feature;  type=random_float;  from=0.1;  to=1;  step=0.01
#key=bagging_fraction;  type=random_float;  from=0.2;  to=1;  step=0.01
#key=bagging_freq;  type=random_int;  from=10;  to=100;  step=1
#key=num_leaves;  type=random_int;  from=16;  to=4096;  step=1
#key=tree_learner;  type=random_from_set;  set='serial','feature','data','voting'
#key=min_data;  type=random_int;  from=1;  to=200;  step=5
#key=max_bin;  type=random_int;  from=2;  to=512;  step=1
#key=min_data_in_bin;  type=random_int;  from=3;  to=10;  step=1
#key=min_gain_to_split;  type=random_float;  from=0.0;  to=1;  step=0.02
#key=feature_fraction_seed;  type=random_int;  from=1;  to=10;  step=1
#key=bagging_seed;  type=random_int;  from=1;  to=10;  step=1
#key=boost_from_average;  type=random_from_set;  set=True,False
#key=is_unbalance;  type=random_from_set;  set=True,False
#key=lambda_l1;  type=random_float;  from=0;  to=1;  step=0.01
#key=lambda_l2;  type=random_float;  from=0;  to=1;  step=0.01
#key=binary_balancing;  type=random_from_set;  set=False
#key=binary_balancing_0;  type=random_float;  from=0.1;  to=1;  step=0.02
#key=binary_balancing_1;  type=random_float;  from=0.1;  to=1;  step=0.02
#key=binary_eval_fun;  type=random_from_set;  set='ROCAUC','PRCAUC'
#key=start_fold;  type=random_from_set;  set=0
#key=max_depth;  type=random_int;  from=-1;  to=10;  step=1
#key=num_round;  type=random_int;  from=100;  to=1500;  step=50
#key=num_threads;  type=random_int;  from=4;  to=4;  step=1
#key=use_float32_dtype; type=random_from_set;  set=True
#key=min_perf_criteria;  type=random_float;  from=0.6;  to=0.6;  step=0.1
#key=shap_data_limit;  type=random_int;  from=25000;  to=25000;  step=1
#key=shap_tree_limit;  type=random_int;  from=-1;  to=-1;  step=1
#key=shap_save_rem; type=random_from_set;  set=False
#key=shap_save_valid; type=random_from_set;  set=False
#key=print_to_html; type=random_from_set;  set=True
#key=print_tables; type=random_from_set;  set=True
#key=out_file_extension;  type=random_from_set;  set=.csv.bz2
#end_of_genes_definitions

# AIOS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

import warnings
warnings.filterwarnings("ignore")

import pandas   as pd
import random   as rn
import lightgbm as lgb
import numpy    as np

import math, json, shap
import os.path, bz2, pickle, joblib
import dateutil
import calendar

from datetime import datetime
from math     import sqrt

from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, log_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix, f1_score, auc
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

import gc
gc.collect()

class cls_ev_agent_{id}:
    # obtain a unique ID for the current instance
    result_id = {id}
    
    # obtain random field (same for all instances within the evolution) which will be the prediction target for this instance/evolution
    target_definition  = "{field_to_predict}"
    out_file_extension = "{out_file_extension}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    target_col  = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]
    
    # create new field name based on "field_ev_prefix" with unique instance ID
    # and filename to save new field data
    field_ev_prefix                  = "{field_ev_prefix}"
    field_ev_prefix_use_source_names = {field_ev_prefix_use_source_names}
    field_ev_prefix_use_target_name  = {field_ev_prefix_use_target_name}
    
    # obtain random selection of fields; number of fields to be selected specified in data:length gene for this instance
    data_defs     = {data}
    fields_to_use = {fields_to_use}
    start_fold    = {start_fold}
    nfolds        = {nfolds}
    rn_seed_init  = {random_seed_init}
    
    params         = {}         # all parameters
    params['algo'] = {}         # ML algo parameters
    dicts_agent    = {}         # various dictionary to be saved as part of model

    params['random_valid']       = {random_valid}
    params['random_valid_size']  = {random_valid_size}
    params['random_valid_folds'] = {random_valid_folds}
    params['random_folds']       = {random_folds}
    params['random_folds_size']  = {random_folds_size}
    params['binary_balancing']   = {binary_balancing}
    params['binary_balancing_0'] = {binary_balancing_0}
    params['binary_balancing_1'] = {binary_balancing_1}    
    params['binary_eval_fun']    = {binary_eval_fun}
    
    # if filter columns are specified then training and validation sets will be selected based on filter criteria
    # based on filter criteria training + validation sets will not necessarily constitute all data, the remainder will be called "test set"
    filter_column   = "{filter_column}"
    filter_column_2 = "{filter_column_2}"
    
    # fields matching the specified prefix will not be used in the model
    ignore_columns_containing  = "{ignore_columns_containing}"
    # include only fields matching string e.g., only properly scaled columns should be used with MLP
    include_columns_containing = "{include_columns_containing}"
    
    objective_multiclass     = {objective_multiclass}
    objective_regression     = {objective_regression}

    models_to_save           = {models_to_save}
    models_apply_on_all_data = {models_apply_on_all_data}

    print_tables         = {print_tables}
    print_to_html        = {print_to_html}
    
    use_validation_set   = {use_validation_set}
    use_float32_dtype    = {use_float32_dtype}
    min_perf_criteria    = {min_perf_criteria}
    
    shap_data_limit      = {shap_data_limit}
    shap_tree_limit      = {shap_tree_limit}
    shap_save_rem        = {shap_save_rem}
    shap_save_valid      = {shap_save_valid}

    def __init__(self):     
        self.set_seed(self.rn_seed_init)        # set same seed for every run of this agent's instance
        
        # remove the target field for this instance from the data used for training
        self.data_defs = [field for field in self.data_defs if field != self.target_definition]

        if self.field_ev_prefix_use_target_name:
            self.field_ev_prefix = self.field_ev_prefix + '_' + self.target_col

        # create new field name based on "field_ev_prefix" with unique instance ID
        # and filename to save new field data      
        if self.field_ev_prefix_use_source_names:                   
            # concatenate all source column names into new field prefix
            col_max_length = int(200 / self.fields_to_use)             # allow 200 characters max combined col name length
            for i in range(0,self.fields_to_use):
                col_name = self.data_defs[i].split("|")[0]
                col_name = col_name[:col_max_length]                   # only take first col_max_length chars from each column
                self.field_ev_prefix = self.field_ev_prefix + '_' + col_name
        
        self.output_column   = self.field_ev_prefix + '_' + str(self.result_id)
        self.output_filename = self.output_column + self.out_file_extension
        
        self.model_env_init()
        
        sfile = workdir + self.output_column + '.model.bz2'
        if os.path.isfile(sfile):
            self.dicts_agent = joblib.load(sfile)
            self.predictors  = self.dicts_agent['models_saved']
            print (str(datetime.now()), self.output_column + ' dictionaries and models loaded')
                
        # obtain columns definitions to filter data set by
        if self.is_set(self.filter_column):
            self.filter_filename = self.filter_column.split("|")[1]
            self.filter_column   = self.filter_column.split("|")[0]
      
        if self.is_set(self.filter_column_2):
            self.filter_filename_2 = self.filter_column_2.split("|")[1]
            self.filter_column_2   = self.filter_column_2.split("|")[0]
    
    
    def model_env_init(self):                                 
        return None
    
    def model_init(self):
        ml_model = None
        return ml_model
    
    def model_save(self, predictor, file_path):
        # predictor['ml_model'].save_model(file_path)
        joblib.dump(predictor, file_path)
        
    def model_load(self, file_path):
        # predictor = {}
        # predictor['ml_model'] = lgb.Booster(model_file=file_path)   
        predictor = joblib.load(file_path)
        return predictor

    def model_feature_importance(self, predictor, n_top_features=25, col_idx=0, importance_type='gain', feat_names=[], print_table=True, to_html=True, feat_map=None):
        importance = predictor.feature_importance(importance_type=importance_type).round(2)
        features   = predictor.feature_name()
        # join field names and their importance values
        col_name = 'Importance_' + str(col_idx)
        fi = pd.DataFrame({'Feature': features, col_name: importance})
        fi[col_name] = round(fi[col_name],4)

        if feat_map != None:
            fi['Feature'] = fi['Feature'].map(feat_map)
                    
        if col_idx == 1:
            self.fi_total = fi
        else:
            self.fi_total = pd.merge(self.fi_total, fi, how='outer', on='Feature', sort=False)

        if print_table:
            print ()
            self.print_html(fi[fi[col_name]>0].sort_values(by=[col_name], ascending=False), max_rows=n_top_features * 2, max_cols=2)

    def model_train(self, ml_model, x_train, y_train, x_test, y_test, current_fold):        
        print (str(datetime.now()), " ML model Training")
       
        x_train    =  lgb.Dataset(x_train, label=y_train)    # convert DF to lgb.Dataset as required by LGBM            
        watchlist  = [lgb.Dataset(x_test,  label=y_test)]
                    
        if self.is_binary and self.params['binary_eval_fun'] == 'PRCAUC':
            ml_model = lgb.train( self.params['algo'], x_train, self.params['algo']['num_round'], watchlist, feval=self.f_eval_prc_auc)
        else:
            ml_model = lgb.train( self.params['algo'], x_train, self.params['algo']['num_round'], watchlist)          
    
        self.model_feature_importance(ml_model, n_top_features=25, col_idx=current_fold, importance_type='gain', print_table=self.print_tables, to_html=self.print_to_html)

        return {'ml_model':ml_model}

    def model_predict(self, predictor, xt, fold=-1, mode=0):
        try:            
            print (str(datetime.now()), " ML model predict data")
            pred = predictor['ml_model'].predict(xt)
        except Exception as e:
            print ('lGBM Predict error: ', e)
            pred = 0

        return pred

    def model_params(self):    
        self.params['algo']['boosting_type']                = {boosting_type}
        self.params['algo']['learning_rate']                = {learning_rate}
        self.params['algo']['sub_feature']                  = {sub_feature}

        self.params['algo']['bagging_fraction']             = {bagging_fraction}
        self.params['algo']['bagging_freq']                 = {bagging_freq}
        self.params['algo']['num_leaves']                   = {num_leaves}
        self.params['algo']['tree_learner']                 = {tree_learner}

        self.params['algo']['min_data']                     = {min_data}
        self.params['algo']['max_bin']                      = {max_bin}
        
        self.params['algo']['min_data_in_bin']              = {min_data_in_bin}
        self.params['algo']['min_gain_to_split']            = {min_gain_to_split}
        
        self.params['algo']['feature_fraction_seed']        = {feature_fraction_seed}
        self.params['algo']['bagging_seed']                 = {bagging_seed}

        self.params['algo']['max_depth']                    = {max_depth}
        self.params['algo']['num_round']                    = {num_round}
        self.params['algo']['early_stopping_rounds']        = 100
        
        self.params['algo']['boost_from_average']           = {boost_from_average}
        self.params['algo']['is_unbalance']                 = {is_unbalance}
        self.params['algo']['lambda_l1']                    = {lambda_l1}
        self.params['algo']['lambda_l2']                    = {lambda_l2}
        
        self.params['algo']['random_seed']                  = self.rn_seed_init     
        self.params['algo']['num_threads']                  = {num_threads}
        self.params['algo']['verbose']                      = -1
        
        if self.is_binary:
            print ("detected binary target: use AUC/LOGLOSS and Binary Cross Entropy loss evaluation")
            self.params['algo']['objective']    = 'binary'
            
            if self.params['binary_eval_fun'] == 'PRCAUC':
                self.params['algo']['metric']   = ['prc_auc', 'auc', 'binary_logloss']
            else:
                self.params['algo']['metric']   = ['auc', 'binary_logloss']
           
            self.params['algo']['num_class']            = 1
            #self.params['algo']['prediction_type']      = 'Probability'
 
        elif self.is_set(self.objective_multiclass):
            print ("detected multi-class target: use Multi-LogLoss/Error; " + str(len(self.target_classes)) + " classes")
            self.params['algo']['objective']    = self.objective_multiclass
            self.params['algo']['metric']       = ['multi_logloss','multi_error']
            
            self.params['algo']['num_class']            = int(max(self.target_classes) + 1)  # requires all int numbers from 0 to max to be classes
            #self.params['algo']['prediction_type']      = 'Probability'

        else:
            print ("detected regression target: use RMSE/MAE")
            self.params['algo']['objective']    = self.objective_regression
            self.params['algo']['metric']       = ['rmse','mae']
            
            self.params['algo']['num_class']            = 1
            #self.params['algo']['prediction_type']      = 'RawFormulaVal'

    
    def is_set(self, s):
        try:
            not_empty = (len(s)>0 and s!="0")
        except:
            not_empty = True
        return not_empty

    def set_seed(self, seed_init):
        rn.seed(seed_init)
        np.random.seed(seed_init)

    def is_use_column(self, s):
        # AIOS Kernel now selects columns using agent parameters
        # so no need to filter inside the agent
        return True
        
    def timestamp(self, x):
        return calendar.timegm(dateutil.parser.parse(x).timetuple())
     
    def print_tbl(self, mesg):
        if self.print_tables:
            print (mesg)
    
    def print_html(self, df, max_rows=50, max_cols=25, jup_notebook=True):
        if self.print_to_html:
            print (df.to_html(max_rows=max_rows,max_cols=max_cols))
        elif jup_notebook:
            pd.set_option("display.min_rows", max_rows)
            pd.set_option("display.max_rows", max_rows)
            pd.set_option("display.max_columns", max_cols)
            display (df)
        else:
            print (df)

    def list_mean(self, lst, precision=4):
        return np.round(sum(lst)/float(len(lst)), decimals=precision)
    
    def f_eval_prc_auc(self, pred, train_data):   
        precision, recall, thresholds = precision_recall_curve(train_data.get_label(), pred)  
        prc_auc = auc(recall, precision)
                     
        return 'prc_auc', prc_auc, True
                     
    def prc_auc(self, train_y, pred):  
        precision, recall, thresholds = precision_recall_curve(train_y, pred)  
        prc_auc = auc(recall, precision)
                     
        return prc_auc 

    def load_columns(self, map_dict=False):
        # start from loading the target field
        df_all = pd.read_csv(workdir+self.target_file, usecols=[self.target_col])[[self.target_col]]

        columns_uniq = [self.target_col]
        columns_orig = [self.target_col]

        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        print (str(datetime.now()), " start loading data")
        block_progress = 0
        block          = int(self.fields_to_use/20)

        for i in range(0,self.fields_to_use):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if self.is_use_column(col_name):
                df_col = pd.read_csv(workdir+file_name, usecols=[col_name])[[col_name]]       # read column from csv file
                
                # if column has associated dictionary csv then it's a text column
                # add dict_ column with actual text
                dict_file_name = workdir+'dict_'+file_name
                if os.path.isfile(dict_file_name) and map_dict:
                    col_name_add = 'dict_'+col_name

                    dict1 = pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()   # load dictionary
                    df_col[col_name_add] = df_col[col_name].map(dict1)                                                 # map and replace
                    #self.dicts_agent[col_name] = dict1                        # save in dictionary of dictionaries to be saved with model files
                    
                    df_col[col_name_add] = df_col[col_name_add].astype(str).apply(self.clean_text)
                else:
                    col_name_add = col_name
                    if df_col[col_name].dtype == np.float64 and self.use_float32_dtype:           # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(np.float32)
                    
                columns_orig.append(col_name_add)
                df_all = df_all.merge(df_col[[col_name_add]], left_index=True, right_index=True)  # add column to the overall dataframe
                
                block_progress += 1
                if (block_progress >= block):
                    block_progress = 0
                    print (str(datetime.now()), " data loaded: ", round((i+1)/self.fields_to_use*100,0), "%")

                # some columns may appear multiple times in data_defs as inherited from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                
                ncol_count = columns_orig.count(col_name_add)
                if ncol_count == 1:
                    columns_uniq.append(col_name_add)
                else:
                    columns_uniq.append(col_name_add + '_v' + str(ncol_count))

        # save actual non-unique columns needed for building the model
        self.dicts_agent['columns_needed'] = columns_orig

        # rename columns in df to unique names
        df_all.columns = columns_uniq
        print (str(datetime.now()), " data loaded", len(df_all), "rows; ", len(df_all.columns), "columns as follows: ")
        print (df_all.columns)
        
        return df_all

 
    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        global dicts

        # by this stage all text fields should be supplied as dict_ fields by previous agents that created such fields in AIOS
        # this agent works with numerical fields only
        columns_uniq = []
        columns_orig = []

        # assemble columns from a list saved during training
        first_col = True
        for col_name in self.dicts_agent['columns_needed']:  
            if col_name != self.target_col:          
                # assemble dataframe column by column             
                df_col = df_add[[col_name]]

                if col_name.startswith('dict_'):                
                    df_col[col_name] = df_col[col_name].astype(str).apply(self.clean_text)    
                else:
                    if df_col[col_name].dtype == np.float64 and self.use_float32_dtype:
                        df_col[col_name] = df_col[col_name].astype(np.float32)
                            
                if first_col:
                    df = df_col[[col_name]]
                    first_col = False
                else:
                    df = df.merge(df_col[[col_name]], left_index=True, right_index=True)
                    
                # some columns may appear multiple times in data_defs as inhereted from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                columns_orig.append(col_name)
                ncol_count = columns_orig.count(col_name)
                if ncol_count == 1:
                    columns_uniq.append(col_name)
                else:
                    columns_uniq.append(col_name + '_v' + str(ncol_count))
        
        # rename columns in df to unique names
        df.columns = columns_uniq
        
        # predict new data set in df applying model for each fold used for training
        pred = np.zeros(len(df))
        if self.dicts_agent['params']['algo']['objective'] == self.objective_multiclass:
            # create a list of lists depending on number of classes used for training 
            # as each prediction is a list of values against each class
            pred = [np.zeros(self.dicts_agent['params']['algo']['num_class']) for i in range(len(df))]
         
        # apply model from each fold created during training and sum their predictions
        for fold in range(0, len(self.predictors)):
            pred += self.model_predict(self.predictors[fold], df)

        if self.dicts_agent['params']['algo']['objective'] == self.objective_multiclass:
            # list of columns for each class probability output
            pred_prob_cols = []   
            for i in range(self.dicts_agent['params']['algo']['num_class']):
                df_add[self.output_column+'_proba_'+str(i)] = 0
                pred_prob_cols.append(self.output_column+'_proba_'+str(i))

            df_add[pred_prob_cols] = pred

            # select class with largest total value in case of multiclass
            pred = np.argmax(pred, axis=1)
        else:
            # average prediction over all folds in case of binary or regression
            pred = pred / len(self.predictors)
        
        df_add[self.output_column] = pred
        

    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        print ("enter run mode " + str(mode))  # 0=work for fitness only;  1=make new output field
                 
        # obtain indexes for train and remainder sets
        # load target column as it may be needed for filtering and removing NaN targets from training
        df_filter_column       = pd.read_csv(workdir+self.target_file, usecols=[self.target_col])
        filter_condition_train = df_filter_column[self.target_col].notnull()
        
        # applying specified filters
        if self.is_set(self.filter_column):
            # load columns to filter by
            df_t = pd.read_csv(workdir+self.filter_filename, usecols = [self.filter_column])
            df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)
            
            filter_condition_train = np.logical_and( filter_condition_train,
                                        np.logical_and( df_filter_column[self.filter_column]>={train_set_from},       
                                                             df_filter_column[self.filter_column]<{train_set_to} ) )  
            
            # two filter columns specified
            if self.is_set(self.filter_column_2):
                df_t = pd.read_csv(workdir+self.filter_filename_2, usecols = [self.filter_column_2])
                df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)
                
                condition2 = np.logical_and( df_filter_column[self.filter_column_2]>={train_set_from_2},                  
                                             df_filter_column[self.filter_column_2]<{train_set_to_2} )                   
                filter_condition_train = np.logical_and( filter_condition_train, condition2 ) 
            
        train_filtered_indexes = df_filter_column[filter_condition_train].index.tolist()
        remainder_set_indexes  = df_filter_column[np.logical_not(filter_condition_train)].index.tolist()   # remainder which is not in train
        
        # initialise prediction column for entire data set as it will be aggregate prediction from multiple folds
        df_filter_column[self.output_column+'_folds_pred']       = 0
        df_filter_column[self.output_column+'_folds_pred_count'] = 0   # number of predictions for each record as different folds will predict different records, so each record may have unique number of predictions
  
        # load specified in data_defs colums of data up-to fields_to_use quantity
        df_all = self.load_columns(map_dict=False)       
        original_row_count = len(df_all)

        # analyse target column whether it is binary which may result in different loss function used
        self.target_classes = df_all[df_all[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()
        self.is_binary      = self.target_classes==[0, 1]
        
        self.params['input_dim'] = len(df_all.columns) - 1  # need this for some models init; df.columns includes the target column, hence need to do -1

        # configure ML model specific parameters which will be saved in self.params dictionary
        self.model_params()

        # initialise temp df holding multi-class predictions for entire data set
        df_filter_column_mc = pd.DataFrame([np.zeros(self.params['algo']['num_class']) for i in range(len(df_filter_column))])

        self.dicts_agent['params']   = self.params
            
        train_sets_ix                = []      # indexes of each whole set used for training
        valid_sets_ix                = []
        train_sub_sets_ix            = []      # indexes of each subset of whole set used for training
        test_sub_sets_ix             = []      # indexes of each subset of whole set used for out-of-sample testing during training
        predictors_all               = []
        weighted_result_folds        = []
        weighted_auc_folds           = []
        valid_result_folds           = []
        valid_result_auc_folds       = []
        valid_set_shap_values        = None
        
        fold_all = 0
        # repeat cross-validation multiple times with different validation set each time
        for valid_fold in range(0, self.params['random_valid_folds']):
            print ()
            print (str(datetime.now())," ----- VALID FOLD: ", valid_fold)
            # obtain indexes for validation set if required
            # applying specified filters

            # assemble condition for filtering validation set
            filter_condition_valid = df_filter_column[self.target_col].notnull()

            if self.is_set(self.filter_column):
                filter_condition_valid = np.logical_and( filter_condition_valid,
                                            np.logical_and( df_filter_column[self.filter_column]>={valid_set_from},           
                                                                    df_filter_column[self.filter_column]<{valid_set_to} ) )    
                # two filter columns specified
                if self.is_set(self.filter_column_2):
                    condition2 = np.logical_and( df_filter_column[self.filter_column_2]>={valid_set_from_2},                   
                                                    df_filter_column[self.filter_column_2]<{valid_set_to_2} )                                
                    filter_condition_valid = np.logical_and( filter_condition_valid, condition2 )

            if self.params['random_valid'] == False:
                # select validation based on fixed filter - may intersect with test or remainder set
                train_sets_ix.append( train_filtered_indexes )
                valid_sets_ix.append( df_filter_column[filter_condition_valid].index.tolist() )
            else:
                # apply stratified random selection to previously filtered train set
                y   = df_filter_column[df_filter_column.index.isin(train_filtered_indexes)][[self.target_col]]
                iy  = y.reset_index(level=0)                                                    # create copy, save existing index in 'index' column and reset index 
                y.reset_index(drop=True, inplace=True)                                          # reset index because StratifiedShuffleSplit will reset index anyway
                    
                if self.is_binary or self.is_set(self.objective_multiclass):
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=self.params['random_valid_size'])

                    for train_ix, valid_ix in sss.split(np.zeros(len(y)), y):
                        train_sets_ix.append( iy[iy.index.isin(train_ix)]['index'].tolist() )       # obtain original indexes from saved copy of labels with original indexes
                        valid_sets_ix.append( iy[iy.index.isin(valid_ix)]['index'].tolist() )       # can't use train_ix, valid_ix directly because they refer to new index reset during shuffling

                else:
                    train_y, valid_y = train_test_split(iy, test_size=self.params['random_valid_size'])
                    train_sets_ix.append( train_y['index'].tolist() )       # obtain original indexes from saved copy of labels with original indexes
                    valid_sets_ix.append( valid_y['index'].tolist() )       # train_test_split produces data sets, so just access previously saved column with indexes
                        
                print ('TRAIN target mean: ', round(df_filter_column[df_filter_column.index.isin(train_sets_ix[valid_fold])][self.target_col].mean(),3))
                print ('VALID target mean: ', round(df_filter_column[df_filter_column.index.isin(valid_sets_ix[valid_fold])][self.target_col].mean(),3))
                    
            # save indexes used for splits
            self.dicts_agent['train_sets_ix']    = train_sets_ix
            self.dicts_agent['remainder_set_ix'] = remainder_set_indexes
            self.dicts_agent['valid_sets_ix']    = valid_sets_ix

            print ("Length of train set         : ", len(train_sets_ix[valid_fold]))
            print ("Length of validation set    : ", len(valid_sets_ix[valid_fold]))
            print ("Length of test/remainder set: ", len(remainder_set_indexes))
            

            # duplicate originally loaded data
            df        = df_all.copy()

            # use previously calculated indexes to select train, validation and remainder sets
            df_test   = df[df.index.isin(remainder_set_indexes)]
    
            df_valid  = df[df.index.isin(valid_sets_ix[valid_fold])]

            # initialise prediction column for validation as it will be aggregate prediction from multiple folds
            predicted_valid_set = np.zeros(len(df_valid))

            # Multi-class case: initialise prediction list of lists depending on number of classes 
            # as each prediction is a list of values against each class
            if self.params['algo']['objective'] == self.objective_multiclass:
                predicted_valid_set = [np.zeros(self.params['algo']['num_class']) for i in range(len(df_valid))]

            # select actual training data for current validation fold
            df = df[df.index.isin(train_sets_ix[valid_fold])]

            # initialise prediction column for main train set as it will be aggregate prediction from multiple training folds
            prediction          = np.zeros(len(df))

            # initialise prediction column for remainder set as it will be aggregate prediction from multiple training folds   
            predicted_test_set  = np.zeros(len(df_test))

            # Multi-class case: initialise prediction list of lists depending on number of classes 
            # as each prediction is a list of values against each class
            if self.params['algo']['objective'] == self.objective_multiclass:
                prediction         = [np.zeros(self.params['algo']['num_class']) for i in range(len(df))]
                predicted_test_set = [np.zeros(self.params['algo']['num_class']) for i in range(len(df_test))]           

            #############################################################
            #         MAIN TRAINING DATA Cross Validation LOOP
            #############################################################

            weighted_result       = 0
            weighted_auc          = 0
            count_records_notnull = 0

            # models for the current validation fold
            predictors            = []

            #obtain target values
            y   = df[[self.target_col]]

            # create copy, save existing index in 'index' column and reset index
            iy  = y.reset_index(level=0)        

            # reset index because StratifiedShuffleSplit will reset index anyway                                             
            y.reset_index(drop=True, inplace=True)                                          

            # randomly select train and test sets from the overall training data set
            for test_fld in range(0, self.nfolds):
                fold_all += 1
                print ()
                print (str(datetime.now())," Train/Test FOLD: ", fold_all)

                if self.is_binary or self.is_set(self.objective_multiclass):
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=self.params['random_folds_size'])
                    for train_ix, test_ix in sss.split(np.zeros(len(y)), y):
                        # obtain original indexes from saved copy of labels with original indexes
                        # can't use train_ix, test_ix directly because they refer to new index reset during shuffling
                        train_ix_orig = iy[iy.index.isin(train_ix)]['index'].tolist()    
                        test_ix_orig  = iy[iy.index.isin(test_ix)]['index'].tolist()        
                else:
                    # use train_test_split for regression tasks with non-stratified shuffling
                    train_y, test_y = train_test_split(iy, test_size=self.params['random_folds_size'])  

                    # obtain original indexes from saved copy of labels with original indexes
                    # train_test_split produces data sets, so just access previously saved column with indexes
                    train_ix_orig   = train_y['index'].tolist()       
                    test_ix_orig    =  test_y['index'].tolist()        
                    
                #------ balance train set -----------------------------------------------------------------------------------------------------
                if self.is_binary and self.params['binary_balancing']:                                           
                    bal_y    = df[[self.target_col]]
                    
                    bal_cond = np.logical_and( bal_y.index.isin(train_ix_orig), bal_y[self.target_col]==0 )                                      
                    train_ix_orig_balanced_0 = bal_y[bal_cond].index.tolist()
                    train_balanced_size_0    = int(len(train_ix_orig_balanced_0) * self.params['binary_balancing_0'])
                    train_ix_orig_balanced_0 = np.random.choice(train_ix_orig_balanced_0, train_balanced_size_0, replace=False).tolist()

                    bal_cond = np.logical_and( bal_y.index.isin(train_ix_orig), bal_y[self.target_col]==1 )
                    train_ix_orig_balanced_1 = bal_y[bal_cond].index.tolist()
                    train_balanced_size_1    = int(len(train_ix_orig_balanced_1) * self.params['binary_balancing_1'])
                    train_ix_orig_balanced_1 = np.random.choice(train_ix_orig_balanced_1, train_balanced_size_1, replace=False).tolist()

                    train_ix_orig = train_ix_orig_balanced_0 + train_ix_orig_balanced_1
                #------------------------------------------------------------------------------------------------------------------------------

                # save indexes in the overall list for all folds      
                train_sub_sets_ix.append(train_ix_orig)                             
                test_sub_sets_ix.append(test_ix_orig)                      

                x_test  = df[df.index.isin(test_ix_orig)]
                x_train = df[df.index.isin(train_ix_orig)]

                print ("x_test  rows count: " + str(len(x_test)))
                print ("x_train rows count: " + str(len(x_train)))

                y_train = x_train[self.target_col]                    # separate training fields and the target
                x_train = x_train.drop(self.target_col, axis=1)

                y_test = x_test[self.target_col]
                x_test = x_test.drop(self.target_col, axis=1)
                
                print ('Y_TRAIN Target mean: ', round(y_train.mean(),3))
                print ('Y_TEST  Target mean: ', round(y_test.mean(), 3))
                
                predictor = self.model_init()
                predictor = self.model_train(predictor, x_train, y_train, x_test, y_test, fold_all)
                pred      = self.model_predict(predictor, x_test)
            
                try:     
                    y_test = np.asarray(y_test)
                    if self.is_binary:
                        result = log_loss(y_test, pred)
                        # show various metrics
                        result_roc_auc = roc_auc_score(y_test, pred)
                        result_prc_auc = self.prc_auc(y_test, pred)
                        result_mcc     = matthews_corrcoef(y_test, np.asarray(pred)>0.5)

                        print ("ROC AUC score: ", result_roc_auc)
                        print ("PRC AUC score: ", result_prc_auc)
                        print ("MatthCC score: ", result_mcc)

                        if self.print_tables:
                            # assume 0.5 probability threshold
                            result_cm = pd.DataFrame(confusion_matrix(y_test, np.asarray(pred)>0.5))
                            result_cr = pd.DataFrame(classification_report(y_test, np.asarray(pred)>0.5, output_dict=True)).transpose()                           
                            print ("\nConfusion Matrix:")
                            self.print_html(result_cm)
                            print ("\nClassification Report:")
                            self.print_html(result_cr.round(2))

                        # assign predictions to corresponding test records only
                        # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                        df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred']       += pred
                        df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred_count'] += 1

                    elif self.params['algo']['objective'] == self.objective_multiclass:                           
                        pred_classes      = np.argmax(pred, axis=1)
                        result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                        result_acc_score  = accuracy_score(y_test, pred_classes)
                        result_mcc        = matthews_corrcoef(y_test, pred_classes)
                        
                        print ("Precision score: ", result_prec_score)
                        print ("Accuracy  score: ", result_acc_score)
                        print ("MatthCC   score: ", result_mcc)
                                                    
                        if self.print_tables:
                            result_cm = pd.DataFrame(confusion_matrix(y_test, pred_classes))
                            result_cr = pd.DataFrame(classification_report(y_test, pred_classes, output_dict=True)).transpose() 
                            print ("\nConfusion Matrix:")
                            self.print_html(result_cm)
                            print ("\nClassification Report:")
                            self.print_html(result_cr.round(2))

                        result         = predictor['ml_model'].best_score['valid_0']['multi_logloss']
                        result_roc_auc = f1_score(y_test, pred_classes, average='weighted')

                        # assign predictions to corresponding test records only
                        # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking

                        # get array of previous folds test records predictions
                        df_pred = np.array(df_filter_column_mc.loc[test_ix_orig])                     
                        df_pred += pred
                        
                        # temp df holding multi-class prediction
                        df_filter_column_mc.loc[test_ix_orig] = df_pred                               
                        df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred_count'] += 1
                        
                    else:
                        # regression case
                        result         = sum(abs(y_test-pred))/len(y_test)
                        result_roc_auc = r2_score(y_test, pred)
                        #result = sqrt(mean_squared_error(y_test, pred))

                        print ("R squared score: ", result_roc_auc)

                        # assign predictions to corresponding test records only
                        # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                        df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred']       += pred
                        df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred_count'] += 1

                except Exception as e:
                    print (e)
                    result         = 999999
                    result_roc_auc = 0
                                                    
                print (f"Test Fold {fold_all} model performance score: ", result)

                weighted_result       += result * len(pred)
                weighted_auc          += result_roc_auc * len(pred)
                count_records_notnull += len(pred)
                
                if result_roc_auc < self.min_perf_criteria:
                    print ("Minimum performance criteria: " + str(self.min_perf_criteria) + " not met! result_roc_auc: " + str(result_roc_auc))
                    return

                if self.is_binary and self.params['binary_eval_fun'] == 'PRCAUC':
                    predictors.append([predictor, result, result_prc_auc])
                    predictors_all.append([predictor, result ,result_prc_auc])    # add predictors to global list across all validation folds
                else:
                    predictors.append([predictor, result, result_roc_auc])
                    predictors_all.append([predictor, result, result_roc_auc])    # add predictors to global list across all validation folds
            #-------------- end of train test CV loop ---------------------------------------------------------------------------------------------

            predictors = pd.DataFrame(predictors, columns=['predictor','result','result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
            print ('\nFolds Performance Overall:')
            self.print_html( predictors, max_rows=50, max_cols=5 )

            if self.models_to_save == 3:
                predictors['result_roc_auc_mean']      = predictors['result_roc_auc'].mean()
                predictors['result_roc_auc_mean_diff'] = abs(predictors['result_roc_auc'] - predictors['result_roc_auc_mean'])
                
                best_predictor_idx  = predictors['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors['result_roc_auc'].idxmin()
                avg_predictor_idx   = predictors['result_roc_auc_mean_diff'].idxmin()
                
                predictors = [predictors['predictor'][worst_predictor_idx], 
                              predictors['predictor'][avg_predictor_idx], 
                              predictors['predictor'][best_predictor_idx]]
                                                    
                print('Selected predictor ids: ', [worst_predictor_idx, avg_predictor_idx, best_predictor_idx])
            else:
                # use all models
                predictors = predictors['predictor'].to_list()
                print('Selected predictor ids: ', predictors)
            
            #------------------ predict remaining and validation samples --------------------------------------------
            for fold in range(0, len(predictors)):             
                # predict remainder set in the column output mode
                if len(df_test) > 0 and mode==1 and self.models_apply_on_all_data==False:
                    pred = self.model_predict(predictors[fold], df_test.drop(self.target_col, axis=1))
                    predicted_test_set  += pred
                    
                    if self.params['algo']['objective'] == self.objective_multiclass: 
                        # assign predictions to corresponding test records only
                        df_pred = np.array(df_filter_column_mc.loc[remainder_set_indexes])              # get array of previous folds test records predictions
                        df_pred += pred
                        df_filter_column_mc.loc[remainder_set_indexes] = df_pred                        # temp df holding multi-class prediction
                        df_filter_column.loc[remainder_set_indexes, self.output_column+'_folds_pred_count'] += 1
                    else:
                        df_filter_column.loc[remainder_set_indexes, self.output_column+'_folds_pred']       += pred
                        df_filter_column.loc[remainder_set_indexes, self.output_column+'_folds_pred_count'] += 1

                # predict validation set

                df_valid_x = df_valid.drop(self.target_col, axis=1)
                pred       = self.model_predict(predictors[fold], df_valid_x)
                predicted_valid_set += pred
                
                if self.params['algo']['objective'] == self.objective_multiclass: 
                    # assign predictions to corresponding test records only
                    df_pred = np.array(df_filter_column_mc.loc[valid_sets_ix[valid_fold]])          # get array of previous folds test records predictions
                    df_pred += pred
                    df_filter_column_mc.loc[valid_sets_ix[valid_fold]] = df_pred                    # temp df holding multi-class prediction
                    df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column+'_folds_pred_count'] += 1
                else:
                    df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column+'_folds_pred']       += pred
                    df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column+'_folds_pred_count'] += 1
                    
            # ------------------ end of predicting remaining and validation samples ---------------------------------
                
            if self.params['algo']['objective'] != self.objective_multiclass:
                prediction          = prediction / len(predictors)
                predicted_test_set  = predicted_test_set  / len(predictors)
                predicted_valid_set = predicted_valid_set / len(predictors)

            df_filter_column[self.output_column+'_folds_pred_avg'] = df_filter_column[self.output_column+'_folds_pred'] / df_filter_column[self.output_column+'_folds_pred_count']
            #------------ end of train test CV method selection ---------------------------------------------------------

            weighted_result = weighted_result/count_records_notnull
            weighted_auc    = weighted_auc/count_records_notnull
            
            weighted_result_folds.append(weighted_result)
            weighted_auc_folds.append(weighted_auc)
            
            print(f'\n{self.nfolds} Test Folds weighted summary')
            print("weighted perf:", weighted_result)
            print("weighted  auc:", weighted_auc)

            # if multiclass convert list of lists into list of predicted labels
            if self.params['algo']['objective'] == self.objective_multiclass:             
                predicted_valid_set = np.argmax(predicted_valid_set, axis=1)
                predicted_test_set  = np.argmax(predicted_test_set, axis=1)

            print()
            print("*************  VALIDATION SET RESULTS  *****************")
            print("Length of validation set:", len(predicted_valid_set))

            # validation set may have missing labels (NAN), for metrics calc find subset with proper labels
            df_valid['predicted_valid_set'] = predicted_valid_set
            df_valid = df_valid[df_valid[self.target_col].notnull()]
            #df_valid.reset_index(drop=True, inplace=True)
            
            y_valid             = np.array(df_valid[self.target_col])
            predicted_valid_set = np.array(df_valid['predicted_valid_set'])

            try:
                if self.is_binary:                                      
                    result = log_loss(y_valid, predicted_valid_set)
                    print ("LOGLOSS score: ", result)
                    result_roc_auc = roc_auc_score(y_valid, predicted_valid_set)
                    print ("ROC AUC score: ", result_roc_auc)
                    result_prc_auc = self.prc_auc(y_valid, predicted_valid_set)
                    print ("PRC AUC score: ", result_prc_auc)
                    result_mcc     = matthews_corrcoef(y_valid, predicted_valid_set>0.5)
                    print ("MatthCC score: ", result_mcc)

                    if self.print_tables:
                        # assume 0.5 probability threshold
                        result_cm = pd.DataFrame(confusion_matrix(y_valid, predicted_valid_set>0.5)) 
                        result_cr = pd.DataFrame(classification_report(y_valid, predicted_valid_set>0.5, output_dict=True)).transpose()
                        print ("\nConfusion Matrix:")
                        self.print_html(result_cm)
                        print ("\nClassification Report:")
                        self.print_html(result_cr.round(2))

                    valid_result_folds.append(result)
                    valid_result_auc_folds.append(result_roc_auc)

                elif self.params['algo']['objective'] == self.objective_multiclass:
                    result_prec_score = precision_score(y_valid, predicted_valid_set, average='weighted')
                    result_acc_score  = accuracy_score(y_valid, predicted_valid_set)
                    result_mcc        = matthews_corrcoef(y_valid, predicted_valid_set)

                    print ("Precision score: ", result_prec_score)
                    print ("Accuracy  score: ", result_acc_score)
                    print ("MatthCC   score: ", result_mcc)

                    if self.print_tables:
                        result_cm = pd.DataFrame(confusion_matrix(y_valid, predicted_valid_set))
                        result_cr = pd.DataFrame(classification_report(y_valid, predicted_valid_set, output_dict=True)).transpose()
                        print ("\nConfusion Matrix:")
                        self.print_html(result_cm)
                        print ("\nClassification Report:")
                        self.print_html(result_cr.round(2))

                    result         = 1 - result_prec_score
                    result_roc_auc = f1_score(y_valid, predicted_valid_set, average='weighted')
                    
                    valid_result_folds.append(result)
                    valid_result_auc_folds.append(result_roc_auc)
                    
                else:
                    result = mean_absolute_error(y_valid, predicted_valid_set)
                    print ("MAE      : ", result)
                    result_rmse = sqrt(mean_squared_error(y_valid, predicted_valid_set))
                    print ("RMSE     : ", result_rmse)
                    result_roc_auc = r2_score(y_valid, predicted_valid_set) 
                    print ("R Squared: ", result_roc_auc)

                    valid_result_folds.append(result)
                    valid_result_auc_folds.append(result_roc_auc)

            except Exception as e:
                print (e)
                return  # no point to carry on with more folds
                                                    
            print ("\n************* END of VALIDATION SET RESULTS  ****************\n")
        #----------- end of validation sets loop --------------------------------------------------------------------
        
        print ('\nTrain/Valid Folds Predictor Performance Overall:')
        predictors_all = pd.DataFrame(predictors_all, columns=['predictor','result','result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
        self.print_html( predictors_all, max_rows=50, max_cols=5 )
                
        # combine feature importance results from all folds into one table
        fi_cols = [col for col in self.fi_total.columns if 'Importance' in col] 
        self.fi_total['Importance_AVG']      = np.round(self.fi_total[fi_cols].sum(axis=1)/fold_all, decimals=2)  
        
        imp_avg_sum = self.fi_total['Importance_AVG'].sum(axis=0)
        if imp_avg_sum != 0:
            self.fi_total['Importance_AVG_perc'] = round(100 * self.fi_total['Importance_AVG'] / imp_avg_sum, 2)
        else:
            self.fi_total['Importance_AVG_perc'] = 0

        print ('\nFEATURE Importance Overall:')
        self.print_html( self.fi_total[self.fi_total['Importance_AVG_perc']>0][['Feature','Importance_AVG','Importance_AVG_perc']].sort_values(by=['Importance_AVG'], ascending=False), max_rows=200, max_cols=4 )
           
        # save indexes used for splits
        self.dicts_agent['train_sub_sets_ix'] = train_sub_sets_ix
        self.dicts_agent['test_sub_sets_ix']  = test_sub_sets_ix
        
        # save performance summaries across all validation folds
        self.dicts_agent['fi_total']          = self.fi_total
        
        #############################################################
        #                   OUTPUT
        #############################################################
        fi_total_dict = dict(zip(self.fi_total['Feature'],self.fi_total['Importance_AVG_perc']))
        print ("#feature_importance="+json.dumps(fi_total_dict))
        
        if mode == 1:  
            if self.models_to_save == 3:
                # select 3 models from all train/test/valid folds
                predictors_all['result_roc_auc_mean']      = predictors_all['result_roc_auc'].mean()
                predictors_all['result_roc_auc_mean_diff'] = abs(predictors_all['result_roc_auc'] - predictors_all['result_roc_auc_mean'])
                
                best_predictor_idx  = predictors_all['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors_all['result_roc_auc'].idxmin()
                avg_predictor_idx   = predictors_all['result_roc_auc_mean_diff'].idxmin()
                
                predictors = [predictors_all['predictor'][worst_predictor_idx], 
                              predictors_all['predictor'][avg_predictor_idx], 
                              predictors_all['predictor'][best_predictor_idx]]
                                                    
                print('Selected predictor ids: ', [worst_predictor_idx, avg_predictor_idx, best_predictor_idx])
            else:
                # use all models
                predictors = predictors_all['predictor'].to_list()
                print('Selected predictor ids: ', predictors)
            
            # save models with dictionaries
            self.dicts_agent['models_saved'] = predictors

            # save dictionary of all auxiliry data and params into file
            sfile = workdir + self.output_column + '.model.bz2'
            joblib.dump(self.dicts_agent, sfile) 
    
            if self.models_apply_on_all_data:
                print (str(datetime.now())," --- Applying all models to all data --- ")
                pred = np.zeros(len(df_all))
                if self.params['algo']['objective'] == self.objective_multiclass:
                    # create a list of lists depending on number of classes used for training 
                    # as each prediction is a list of values against each class
                    pred = [np.zeros(self.params['algo']['num_class']) for i in range(len(df_all))]
                                        
                    pred_prob_cols = []   # list of columns for each class probability output
                    for i in range(self.params['algo']['num_class']):
                        df_filter_column_mc[self.output_column+'_proba_'+str(i)] = 0
                        pred_prob_cols.append(self.output_column+'_proba_'+str(i))

                # apply model from each fold created during training and sum their predictions              
                for fold in range(0, len(predictors)):
                    pred += self.model_predict(predictors[fold], df_all.drop(self.target_col, axis=1))

                if self.params['algo']['objective'] == self.objective_multiclass:
                    # output probabilities for each class
                    df_filter_column_mc[pred_prob_cols] = pred                                

                    for col in pred_prob_cols:
                        # save probabilities as features
                        fil = col + self.out_file_extension
                        df_filter_column_mc[[col]].to_csv(workdir+fil)
                        print ("#add_field:"+col+",N,"+fil+","+str(original_row_count))  

                    # select class with largest total value in case of multiclass
                    pred = np.argmax(pred, axis=1)
                else:
                    # average prediction over all folds in case of binary or regression
                    pred = pred / len(predictors)
                
                df_all[self.output_column] = pred
                df_all[[self.output_column]].to_csv(workdir+self.output_filename)
            else:
                # predictions have already been assembled during model folds testing
                # if multiclass convert list of lists into list of predicted labels
                if self.params['algo']['objective'] == self.objective_multiclass:             
                    df_filter_column[self.output_column+'_folds_pred'] = np.argmax(np.array(df_filter_column_mc), axis=1)
                    df_filter_column[self.output_column]               = df_filter_column[self.output_column+'_folds_pred'] 
                    df_filter_column.loc[df_filter_column[self.output_column+'_folds_pred_count']==0,self.output_column] = float('nan')
                else:
                    df_filter_column[self.output_column] = df_filter_column[self.output_column+'_folds_pred'] / df_filter_column[self.output_column+'_folds_pred_count']
        
                df_filter_column[[self.output_column]].to_csv(workdir+self.output_filename)

            # print AIOS command to add newly created fields to AIOS Data Lake register
            print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(original_row_count))
            
            print ("b_fitness=" +str(round(1-self.list_mean(weighted_auc_folds)*self.list_mean(valid_result_auc_folds),4)))
            print ("b_result_1="+str(self.list_mean(weighted_result_folds)))
            print ("b_result_2="+str(self.list_mean(weighted_auc_folds)))
            print ("b_result_3="+str(self.list_mean(valid_result_folds)))
            print ("b_result_4="+str(self.list_mean(valid_result_auc_folds)))
        else:
            # main fitness metric
            print ("fitness="     +str(round(1-self.list_mean(weighted_auc_folds)*self.list_mean(valid_result_auc_folds),4)))
            print ("out_result_1="+str(self.list_mean(weighted_result_folds)))                                        # Log Loss in train/test CV
            print ("out_result_2="+str(self.list_mean(weighted_auc_folds)))                                           # ROC AUC in train/test CV
            print ("out_result_3="+str(self.list_mean(valid_result_folds)))                                           # main fitness on Validation
            print ("out_result_4="+str(self.list_mean(valid_result_auc_folds)))                                       # ROC AUC on Validation


ev_agent_{id} = cls_ev_agent_{id}()
