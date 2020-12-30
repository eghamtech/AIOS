#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=13
#key=fields_to_use;  type=random_int;  from=13;  to=13;  step=1
#key=field_ev_prefix;  type=random_from_set;  set=ev_field_logrr_
#key=field_ev_prefix_use_target_name;  type=random_from_set;  set=True
#key=field_ev_prefix_use_source_names;  type=random_from_set;  set=False
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
#key=lr_penalty;  type=random_from_set;  set='l1','l2','elasticnet','none'
#key=lr_dual;  type=random_from_set;  set=True,False
#key=lr_tol;  type=random_float;  from=0.0001;  to=0.0001;  step=0.0001
#key=lr_C;  type=random_float;  from=0.5;  to=3;  step=0.1
#key=lr_fit_intercept;  type=random_from_set;  set=True,False
#key=lr_intercept_scaling;  type=random_float;  from=0.5;  to=1.5;  step=0.1
#key=lr_class_weight;  type=random_from_set;  set='balanced',None
#key=lr_solver;  type=random_from_set;  set='newton-cg','lbfgs','liblinear','sag','saga'
#key=lr_multi_class;  type=random_from_set;  set='auto','ovr','multinomial'
#key=lr_warm_start;  type=random_from_set;  set=False
#key=lr_l1_ratio;  type=random_float;  from=0.0;  to=1;  step=0.01
#key=scaler_use;  type=random_from_set;  set=True,False
#key=scaler_mean;  type=random_from_set;  set=True,False
#key=scaler_std;  type=random_from_set;  set=True,False
#key=binary_balancing;  type=random_from_set;  set=True,False
#key=binary_balancing_0;  type=random_float;  from=0.1;  to=1;  step=0.02
#key=binary_balancing_1;  type=random_float;  from=0.1;  to=1;  step=0.02
#key=binary_eval_fun;  type=random_from_set;  set='ROCAUC','PRCAUC'
#key=start_fold;  type=random_from_set;  set=0
#key=max_depth;  type=random_int;  from=-1;  to=10;  step=1
#key=num_round;  type=random_int;  from=100;  to=500;  step=50
#key=num_threads;  type=random_int;  from=1;  to=1;  step=1
#key=use_float32_dtype; type=random_from_set;  set=True
#key=min_perf_criteria;  type=random_float;  from=0.6;  to=0.6;  step=0.1
#key=use_thresholds_train; type=random_from_set;  set=True
#key=shap_data_limit;  type=random_int;  from=25000;  to=25000;  step=1
#key=shap_tree_limit;  type=random_int;  from=-1;  to=-1;  step=1
#key=shap_save_rem; type=random_from_set;  set=False
#key=shap_save_valid; type=random_from_set;  set=False
#key=print_to_html; type=random_from_set;  set=True
#key=print_tables; type=random_from_set;  set=True
#key=verbose;  type=random_from_set;  set=0
#key=out_file_extension;  type=random_from_set;  set=.csv.bz2
#end_of_genes_definitions

# AICHOO OS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

import warnings
warnings.filterwarnings("ignore")

import pandas   as pd
import random   as rn
import numpy    as np

import math, json, shap
import os.path, bz2, pickle, joblib
import dateutil
import calendar

from datetime import datetime
from math     import sqrt

from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

    num_threads    = {num_threads}
    rn_seed_init   = {random_seed_init}

    models_to_save           = {models_to_save}
    models_apply_on_all_data = {models_apply_on_all_data}
    
    params         = {}         # all parameters
    params['algo'] = {}         # ML algo parameters
    dicts_agent    = {}         # various dictionary to be saved as part of model
    
    # if filter columns are specified then training and validation sets will be selected based on filter criteria
    # based on filter criteria training + validation sets will not necessarily constitute all data, the remainder will be called "test set"
    filter_column   = "{filter_column}"
    filter_column_2 = "{filter_column_2}"
    # filter_filename = trainfile   # filter columns are in trainfile which must be specified in Constants - deprecated
    
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
    
    shap_data_limit    = {shap_data_limit}
    shap_tree_limit    = {shap_tree_limit}
    shap_save_rem      = {shap_save_rem}
    shap_save_valid    = {shap_save_valid}
    
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
        
        sfile = workdir + self.output_column + '_dicts.model'
        if os.path.isfile(sfile):
            rfile = bz2.BZ2File(sfile, 'r')
            self.dicts_agent = pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.output_column + ' dictionaries model loaded')
            
            # if saved model for the target field already exists then load it from filesystem
            self.predictors = []
            if self.dicts_agent['params']['random_folds'] == False:
                from_fold = self.start_fold
                to_fold   = self.nfolds
            else:
                from_fold = 0
                to_fold   = self.dicts_agent['models_saved']                     # use all saved models to make predictions

            for fold in range(from_fold, to_fold):
                sfile = workdir + self.output_column + "_fold" + str(fold) + ".model"
                if os.path.isfile(sfile):
                    predictor_stored = self.model_load(sfile)
                    self.predictors.append(predictor_stored)
                    print (str(datetime.now()), self.output_column + ' fold ' + str(fold) + ' predictor model loaded')
                
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
        lr_model = LogisticRegression( penalty           = self.params['algo']['penalty'], 
                                       dual              = self.params['algo']['dual'], 
                                       tol               = self.params['algo']['tol'], 
                                       C                 = self.params['algo']['C'], 
                                       fit_intercept     = self.params['algo']['fit_intercept'], 
                                       intercept_scaling = self.params['algo']['intercept_scaling'], 
                                       class_weight      = self.params['algo']['class_weight'], 
                                       random_state      = self.params['algo']['random_state'], 
                                       solver            = self.params['algo']['solver'], 
                                       max_iter          = self.params['algo']['max_iter'], 
                                       multi_class       = self.params['algo']['multi_class'],
                                       verbose           = self.params['algo']['verbose'], 
                                       warm_start        = self.params['algo']['warm_start'], 
                                       n_jobs            = self.params['algo']['n_jobs'], 
                                       l1_ratio          = self.params['algo']['l1_ratio']
                                    )
        
        if self.params['algo']['scaler_use']:
            dtscaler = StandardScaler(with_mean=self.params['algo']['scaler_mean'], with_std=self.params['algo']['scaler_std'])
            ml_model = make_pipeline(dtscaler, lr_model)
        else:
            ml_model = make_pipeline(lr_model)
        
        return ml_model
    
    def model_save(self, predictor, file_path):
        joblib.dump(['ml_model'], file_path)

    def model_load(self, file_path):
        predictor = {}
        predictor['ml_model'] = joblib.load(file_path)            
        return predictor

    def model_feature_importance(self, predictor, n_top_features=25, col_idx=0, importance_type='gain', feat_names=[], round_digits=10, print_table=True, to_html=True):
        importance = predictor.coef_[0]
        features   = feat_names
        # join field names and their importance values
        col_name = 'Importance_' + str(col_idx)
        fi = pd.DataFrame({'Feature': features, col_name: importance})
        fi[col_name] = round(fi[col_name],round_digits)

        if col_idx == 1:
            self.fi_total = fi
        else:
            self.fi_total = pd.merge(self.fi_total, fi, how='outer', on='Feature', sort=False)

        if print_table:
            print ()
            self.print_html(fi.sort_values(by=[col_name], ascending=False), max_rows=n_top_features * 2, max_cols=2)


    def model_train(self, ml_model, x_train, y_train, x_test, y_test, current_fold):        
        print (str(datetime.now()), " ML model Training")

        ml_model.fit(x_train, y_train)        
    
        self.model_feature_importance(ml_model.named_steps['logisticregression'], n_top_features=25, col_idx=current_fold, importance_type='gain', feat_names=x_train.columns, round_digits=10, print_table=self.print_tables, to_html=self.print_to_html)

        return {'ml_model':ml_model}

    def model_predict(self, predictor, xt, fold=-1, mode=0):
        try:            
            print (str(datetime.now()), " ML model predict data")
            pred = predictor['ml_model'].predict_proba(xt)

            if self.is_binary:
                pred = pred[:,1]
        except Exception as e:
            print ('Logistic Regression Predict error: ', e)
            pred = 0

        return pred

    def model_params(self):    
        self.params['algo']['penalty']              = {lr_penalty}
        self.params['algo']['dual']                 = {lr_dual}
        self.params['algo']['tol']                  = {lr_tol}

        self.params['algo']['C']                    = {lr_C}
        self.params['algo']['fit_intercept']        = {lr_fit_intercept}
        self.params['algo']['intercept_scaling']    = {lr_intercept_scaling}
        self.params['algo']['class_weight']         = {lr_class_weight}

        self.params['algo']['solver']               = {lr_solver}
        self.params['algo']['max_iter']             = {num_round}
        
        self.params['algo']['multi_class']          = {lr_multi_class}
        self.params['algo']['warm_start']           = {lr_warm_start}
        
        self.params['algo']['l1_ratio']             = {lr_l1_ratio}
        self.params['algo']['scaler_use']           = {scaler_use}
        self.params['algo']['scaler_mean']          = {scaler_mean}
        self.params['algo']['scaler_std']           = {scaler_std}
        
        self.params['algo']['random_state']         = self.rn_seed_init     
        self.params['algo']['n_jobs']               = self.num_threads
        self.params['algo']['verbose']              = {verbose}
        
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
        
        # determine whether given column should be ignored
#         s = s.replace('%','')           # remove % used for pattern matching as now required to filter column by AIOS itself
        
#        if s.find(self.target_col)>=0:  # ignore columns that contain target_col as they are a derivative of the target
#            return False 
#         # ignore other columns containing specified ignore parameter value
#         if self.is_set(self.ignore_columns_containing) and s.find(self.ignore_columns_containing.replace('%',''))>=0:
#             return False
#         # include all columns if include parameter not specified
#         if not self.is_set(self.include_columns_containing):
#             return True
#         # include columns specified in parameter
#         if self.is_set(self.include_columns_containing) and s.find(self.include_columns_containing.replace('%',''))>=0:
#             return True 
#         # ignore all other columns
#         return False
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
            display (df)
        else:
            print (df)

    def list_mean(self, lst, precision=4):
        return np.round(sum(lst)/float(len(lst)), decimals=precision)
    
    def f_eval_prc_auc(self, pred, train_data):   
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc

        precision, recall, thresholds = precision_recall_curve(train_data.get_label(), pred)  
        prc_auc = auc(recall, precision)
                     
        return 'prc_auc', prc_auc, True
                     
    def prc_auc(self, train_y, pred):  
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc

        precision, recall, thresholds = precision_recall_curve(train_y, pred)  
        prc_auc = auc(recall, precision)
                     
        return prc_auc 

    def load_columns(self, map_dict=False):
        # start from loading the target field
        df_all = pd.read_csv(workdir+self.target_file, usecols=[self.target_col])[[self.target_col]]

        columns_new = [self.target_col]
        columns     = [self.target_col]
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        print (str(datetime.now()), " start loading data")
        block_progress = 0
        block          = int(self.fields_to_use/20)

        for i in range(0,self.fields_to_use):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if self.is_use_column(col_name):
                df_col = pd.read_csv(workdir+file_name, usecols=[col_name])[[col_name]]       # read column from csv file
                
                # if column has associated dictionary csv then it's a text column, replace column with actual text
                dict_file_name = workdir+'dict_'+file_name
                if os.path.isfile(dict_file_name) and map_dict:
                    dict1 = pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()       # load dictionary
                    df_col[col_name] = df_col[col_name].map(dict1)                                                         # map and replace
                    self.dicts_agent[col_name] = dict1                                                                     # save in dictionary of dictionaries to be saved with model files
                    
                    df_col[col_name] = df_col[col_name].astype(str).apply(self.clean_text)
                else:
                    if df_col[col_name].dtype == np.float64 and self.use_float32_dtype:           # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(np.float32)

                df_all = df_all.merge(df_col, left_index=True, right_index=True)                       # add column to the overall dataframe

                block_progress += 1
                if (block_progress >= block):
                    block_progress = 0
                    print (str(datetime.now()), " data loaded: ", round((i+1)/self.fields_to_use*100,0), "%")

                # some columns may appear multiple times in data_defs as inherited from parents DNA
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
        global dicts
        # by this stage all text fields should have been converted to dictionary values by previous agents that created such fields in AIOS
        # since this agent works with text fields, actual text values will be obtained from the global dicts variable
        columns_new = []
        columns     = []
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        for i in range(0,self.fields_to_use):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]
            
            if self.is_use_column(col_name):               
                # assemble dataframe column by column             
                df_col = df_add[[col_name]]
                                              
                if df_col[col_name].dtype == np.float64 and self.use_float32_dtype:                               # downcast to save memory if needed
                    df_col[col_name] = df_col[col_name].astype(np.float32)
                        
                if i==0:
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
        pred = np.zeros(len(df))
        if self.dicts_agent['params']['algo']['objective'] == self.objective_multiclass:
            # create a list of lists depending on number of classes used for training 
            # as each prediction is a list of values against each class
            pred = [np.zeros(self.dicts_agent['params']['algo']['num_class']) for i in range(len(df))]
         
        # apply model from each fold created during training and sum their predictions
        if self.dicts_agent['params']['random_folds'] == False:
            for fold in range(self.start_fold, self.nfolds):
                pred += self.model_predict(self.predictors[fold - self.start_fold], df)

            if self.dicts_agent['params']['algo']['objective'] == self.objective_multiclass:
                # select class with largest total value in case of multiclass
                pred = np.argmax(pred, axis=1)
            else:
                # average prediction over all folds in case of binary or regression
                pred = pred / (self.nfolds - self.start_fold)
        else:
            for fold in range(0, len(self.predictors)):
                pred += self.model_predict(self.predictors[fold], df)

            if self.dicts_agent['params']['algo']['objective'] == self.objective_multiclass:
                # select class with largest total value in case of multiclass
                pred = np.argmax(pred, axis=1)
            else:
                # average prediction over all folds in case of binary or regression
                pred = pred / len(self.predictors)
        
        df_add[self.output_column] = pred
        

    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        print ("enter run mode " + str(mode))  # 0=work for fitness only;  1=make new output field
             
        # prepare all parameters
        self.params['random_valid']       = {random_valid}
        self.params['random_valid_size']  = {random_valid_size}
        self.params['random_valid_folds'] = {random_valid_folds}
        self.params['random_folds']       = {random_folds}
        self.params['random_folds_size']  = {random_folds_size}
        self.params['binary_balancing']   = {binary_balancing}
        self.params['binary_balancing_0'] = {binary_balancing_0}
        self.params['binary_balancing_1'] = {binary_balancing_1}    
        self.params['binary_eval_fun']    = {binary_eval_fun}
              
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
        # applies only in case when params['random_valid'] == True
        for valid_fold in range(0, self.params['random_valid_folds']):
            print ()
            print (str(datetime.now())," ----- VALID FOLD: ", valid_fold)
            # obtain indexes for validation set if required
            # applying specified filters
            if self.use_validation_set:
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

            print ("Length of train set:",          len(train_sets_ix[valid_fold]))
            print ("Length of test/remainder set:", len(remainder_set_indexes))
            print ("Length of validation set:",     len(valid_sets_ix[valid_fold]))


            # duplicate originally loaded data
            df      = df_all.copy()
            # use previously calculated indexes to select train, validation and remainder sets
            df_test = df[df.index.isin(remainder_set_indexes)]

            if self.use_validation_set:        
                df_valid  = df[df.index.isin(valid_sets_ix[valid_fold])]
                # initialise prediction column for validation as it will be aggregate prediction from multiple folds
                predicted_valid_set = np.zeros(len(df_valid))                
                # Multi-class case: initialise prediction list of lists depending on number of classes 
                # as each prediction is a list of values against each class
                if self.params['algo']['objective'] == self.objective_multiclass:
                    predicted_valid_set = [np.zeros(self.params['algo']['num_class']) for i in range(len(df_valid))]

            df      = df[df.index.isin(train_sets_ix[valid_fold])]

            # initialise prediction column for main train set as it will be aggregate prediction from multiple folds
            prediction          = np.zeros(len(df))
            # initialise prediction column for remainder set as it will be aggregate prediction from multiple folds   
            predicted_test_set  = np.zeros(len(df_test))
            # Multi-class case: initialise prediction list of lists depending on number of classes 
            # as each prediction is a list of values against each class
            if self.params['algo']['objective'] == self.objective_multiclass:
                prediction         = [np.zeros(self.params['algo']['num_class']) for i in range(len(df))]
                predicted_test_set = [np.zeros(self.params['algo']['num_class']) for i in range(len(df_test))]           

            #############################################################
            #                   MAIN LOOP
            #############################################################

            weighted_result = 0
            weighted_auc    = 0
            count_records_notnull = 0

            if self.params['random_folds'] == False:
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
                    print ("Fold to predict start", range_start, "; end ", range_end)

                    x_test = df[df.index.isin(range_predict)]
                    x_test.reset_index(drop=True, inplace=True)
                    x_test_orig = x_test.copy()                           # save original test set before removing null values
                    x_test = x_test[x_test[self.target_col].notnull()]    # remove examples that have no proper target label
                    x_test.reset_index(drop=True, inplace=True)

                    x_train = df[df.index.isin(range_predict)==False]
                    x_train.reset_index(drop=True, inplace=True)
                    x_train= x_train[x_train[self.target_col].notnull()]  # remove examples that have no proper target label
                    x_train.reset_index(drop=True, inplace=True)

                    print ("x_test rows count: "  + str(len(x_test)))
                    print ("x_train rows count: " + str(len(x_train)))

                    y_train = x_train[self.target_col]                    # separate training fields and the target
                    x_train = x_train.drop(self.target_col, 1)

                    y_test = x_test[self.target_col]
                    x_test = x_test.drop(self.target_col, 1)

                    print ('Y_TRAIN Target mean: ', round(y_train.mean(),3))
                    print ('Y_TEST  Target mean: ', round(y_test.mean(), 3))
                    
                    predictor = self.model_init()
                    predictor = self.model_train(predictor, x_train, y_train, x_test, y_test, fold-self.start_fold+1)
                    pred      = self.model_predict(predictor, x_test)

                    if mode==1:
                        self.model_save(predictor, workdir + self.output_column + "_fold" + str(fold) + ".model")

                    if self.is_binary:
                        result = log_loss(y_test, pred)
                        # show various metrics as per
                        # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                        result_roc_auc = roc_auc_score(y_test, pred)
                        result_prc_auc = self.prc_auc(y_test, pred)
                        print ("ROC AUC score: ", result_roc_auc)
                        print ("PRC AUC score: ", result_prc_auc)
                        
                        if self.print_tables:
                            result_cm = confusion_matrix(y_test, (pred>0.5))  # assume 0.5 probability threshold
                            result_cr = classification_report(y_test, (pred>0.5))                      
                            print ("Confusion Matrix:\n", result_cm)
                            print ("Classification Report:\n", result_cr)
                                                       
                    elif self.is_set(self.objective_multiclass):
                        pred_classes      = np.argmax(pred, axis=1)
                        result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                        result_acc_score  = accuracy_score(y_test, pred_classes)
                        result_cm         = confusion_matrix(y_test, pred_classes)
                        result_cr         = classification_report(y_test, pred_classes)
                        
                        if self.print_tables:
                            print ("Precision score: ", result_prec_score)
                            print ("Accuracy  score: ", result_acc_score)
                            print ("Confusion Matrix:\n",      result_cm)
                            print ("Classification Report:\n", result_cr)
                        result         = predictor['ml_model'].best_score['valid_0']['multi_logloss']
                        result_roc_auc = f1_score(y_test, pred_classes, average='weighted')
                    else:
                        result         = sum(abs(y_test-pred))/len(y_test)
                        result_roc_auc = r2_score(y_test, pred)
                        #result = sqrt(mean_squared_error(y_test, pred))

                    print ("result: ", result)
                                                                                                             
                    if result_roc_auc < self.min_perf_criteria:
                        print ("Minimum performance criteria: " + str(self.min_perf_criteria) + " not met! result_roc_auc: " + str(result_roc_auc))
                        return

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
                    count_records_notnull += len(pred)

                    # predict all examples in the original test set which may include erroneous examples previously removed
                    pred_all_test = self.model_predict(predictor, x_test_orig.drop(self.target_col, axis=1))

                    if self.params['algo']['objective'] == self.objective_multiclass:
                        prediction[range_start:range_end] = np.argmax(pred_all_test, axis=1)
                    else:
                        prediction[range_start:range_end] = pred_all_test

                    # predict validation and remainder sets examples
                    if self.use_validation_set:
                        predicted_valid_set += self.model_predict(predictor, df_valid.drop(self.target_col, axis=1))
                        predicted_test_set  += self.model_predict(predictor, df_test.drop(self.target_col, axis=1))

                predicted_valid_set = predicted_valid_set / (self.nfolds - self.start_fold)
                predicted_test_set  = predicted_test_set / (self.nfolds - self.start_fold)
            else:
                # select folds using random shuffle and stratify
                y   = df[[self.target_col]]
                iy  = y.reset_index(level=0)                                                    # create copy, save existing index in 'index' column and reset index 
                y.reset_index(drop=True, inplace=True)                                          # reset index because StratifiedShuffleSplit will reset index anyway
    
                predictors = []
                
                for test_fld in range(0, self.nfolds):
                    fold_all += 1
                    print ()
                    print (str(datetime.now())," Train/Test FOLD: ", fold_all)

                    if self.is_binary or self.is_set(self.objective_multiclass):
                        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.params['random_folds_size'])
                        for train_ix, test_ix in sss.split(np.zeros(len(y)), y):
                            train_ix_orig = iy[iy.index.isin(train_ix)]['index'].tolist()       # obtain original indexes from saved copy of labels with original indexes
                            test_ix_orig  = iy[iy.index.isin(test_ix)]['index'].tolist()        # can't use train_ix, test_ix directly because they refer to new index reset during shuffling
                    else:
                        train_y, test_y = train_test_split(iy, test_size=self.params['random_folds_size'])  # use train_test_split for regression tasks with non-stratified shuffling
                        train_ix_orig   = train_y['index'].tolist()        # obtain original indexes from saved copy of labels with original indexes
                        test_ix_orig    =  test_y['index'].tolist()        # train_test_split produces data sets, so just access previously saved column with indexes
                        
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
                                              
                    train_sub_sets_ix.append(train_ix_orig)                             # save indexes in the overall list for all folds
                    test_sub_sets_ix.append(test_ix_orig)                      

                    x_test  = df[df.index.isin(test_ix_orig)]
                    x_train = df[df.index.isin(train_ix_orig)]

                    print ("x_test  rows count: " + str(len(x_test)))
                    print ("x_train rows count: " + str(len(x_train)))

                    y_train = x_train[self.target_col]                    # separate training fields and the target
                    x_train = x_train.drop(self.target_col, 1)

                    y_test = x_test[self.target_col]
                    x_test = x_test.drop(self.target_col, 1)
                    
                    print ('Y_TRAIN Target mean: ', round(y_train.mean(),3))
                    print ('Y_TEST  Target mean: ', round(y_test.mean(), 3))
                   
                    predictor = self.model_init()
                    predictor = self.model_train(predictor, x_train, y_train, x_test, y_test, fold_all)
                    pred      = self.model_predict(predictor, x_test)
                
                    try:     
                        y_test = np.asarray(y_test)
                        if self.is_binary:
                            result = log_loss(y_test, pred)
                            # show various metrics as per
                            # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                            result_roc_auc = roc_auc_score(y_test, pred)
                            result_prc_auc = self.prc_auc(y_test, pred)
                            print ("ROC AUC score: ", result_roc_auc)
                            print ("PRC AUC score: ", result_prc_auc)

                            if self.print_tables:
                                result_cm      = confusion_matrix(y_test, np.asarray(pred)>0.5)  # assume 0.5 probability threshold
                                result_cr      = classification_report(y_test, np.asarray(pred)>0.5)                           
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred']       += pred
                            df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred_count'] += 1

                        elif self.params['algo']['objective'] == self.objective_multiclass:                           
                            pred_classes      = np.argmax(pred, axis=1)
                            result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                            result_acc_score  = accuracy_score(y_test, pred_classes)
                            result_cm = confusion_matrix(y_test, pred_classes)
                            result_cr = classification_report(y_test, pred_classes)
                                                       
                            if self.print_tables:
                                print ("Precision score: ", result_prec_score)
                                print ("Accuracy score: ",  result_acc_score)
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)

                            result         = log_loss(y_test, pred)
                            result_roc_auc = f1_score(y_test, pred_classes, average='weighted')

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_pred = np.array(df_filter_column_mc.loc[test_ix_orig])                     # get array of previous folds test records predictions
                            df_pred += pred
                            df_filter_column_mc.loc[test_ix_orig] = df_pred                               # temp df holding multi-class prediction
                            df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred_count'] += 1
                          
                        else:
                            result         = sum(abs(y_test-pred))/len(y_test)
                            result_roc_auc = r2_score(y_test, pred)
                            #result = sqrt(mean_squared_error(y_test, pred))

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred']       += pred
                            df_filter_column.loc[test_ix_orig, self.output_column+'_folds_pred_count'] += 1

                    except Exception as e:
                        print (e)
                        result         = 999999
                        result_roc_auc = 0
                                                       
                    print ("result: ", result)

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
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

                        if mode == 1 and valid_fold == 0 and fold == len(predictors) - 1 and self.shap_save_rem:
                            # save SHAP values only for the best predictor from the first validation set
                            explainer = shap.LinearExplainer(predictors[fold]['ml_model'].named_steps['logisticregression'])
                            rem_set_shap_values = explainer.shap_values(df_test[0:self.shap_data_limit].drop(self.target_col, axis=1), tree_limit=self.shap_tree_limit)
                                
                            self.dicts_agent['fi_rem_shap'].append(rem_set_shap_values)
                            #self.dicts_agent['fi_rem_x'].append(df_test)
                            self.dicts_agent['fi_rem_y'].append(pred[0:self.shap_data_limit])
                            self.dicts_agent['fi_rem_expl'].append(explainer)

                    # predict validation set
                    if self.use_validation_set:
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

                        if mode == 1 and valid_fold == 0 and fold == len(predictors) - 1 and self.shap_save_valid:
                            # save SHAP values only for the best predictor from the first validation set
                            explainer = shap.LinearExplainer(predictors[fold]['ml_model'].named_steps['logisticregression'])
                            valid_set_shap_values = explainer.shap_values(df_valid_x[0:self.shap_data_limit], tree_limit=self.shap_tree_limit)
                                
                            self.dicts_agent['fi_valid_shap'].append(valid_set_shap_values)
                            self.dicts_agent['fi_valid_x'].append(df_valid_x[0:self.shap_data_limit])
                            self.dicts_agent['fi_valid_y'].append(pred[0:self.shap_data_limit])
                            self.dicts_agent['fi_valid_expl'].append(explainer)
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
            
            print ("\nweighted_result:", weighted_result)
            print ("weighted_auc:",      weighted_auc)

            # if multiclass convert list of lists into list of predicted labels
            if self.params['algo']['objective'] == self.objective_multiclass:             
                predicted_valid_set = np.argmax(predicted_valid_set, axis=1)
                predicted_test_set  = np.argmax(predicted_test_set, axis=1)

            if self.use_validation_set:
                print()
                print ("*************  VALIDATION SET RESULTS  *****************")
                print ("Length of validation set:", len(predicted_valid_set))

                # validation set may have missing labels (NAN), for metrics calc find subset with proper labels
                df_valid['predicted_valid_set'] = predicted_valid_set
                df_valid = df_valid[df_valid[self.target_col].notnull()]
                #df_valid.reset_index(drop=True, inplace=True)
                y_valid             = np.array(df_valid[self.target_col])
                predicted_valid_set = np.array(df_valid['predicted_valid_set'])

                try:
                    if self.is_binary:                                      
                        result = log_loss(y_valid, predicted_valid_set)
                        print ("LOGLOSS: ", result)
                        result_roc_auc = roc_auc_score(y_valid, predicted_valid_set)
                        print ("ROC AUC score: ", result_roc_auc)
                        result_prc_auc = self.prc_auc(y_valid, predicted_valid_set)
                        print ("PRC AUC score: ", result_prc_auc)

                        if self.print_tables:
                            result_cm = confusion_matrix(y_valid, predicted_valid_set>0.5)  # assume 0.5 probability threshold
                            print ("Confusion Matrix:\n",      result_cm)
                            result_cr = classification_report(y_valid, predicted_valid_set>0.5)
                            print ("Classification Report:\n", result_cr)

                        valid_result_folds.append(result)
                        valid_result_auc_folds.append(result_roc_auc)

                    elif self.params['algo']['objective'] == self.objective_multiclass:
                        result_prec_score = precision_score(y_valid, predicted_valid_set, average='weighted')
                        result_acc_score  = accuracy_score(y_valid, predicted_valid_set)
                        result_cm = confusion_matrix(y_valid, predicted_valid_set)
                        result_cr = classification_report(y_valid, predicted_valid_set)

                        if self.print_tables:
                            print ("Precision score: ",        result_prec_score)
                            print ("Accuracy score: ",         result_acc_score)
                            print ("Confusion Matrix:\n",      result_cm)
                            print ("Classification Report:\n", result_cr)

                        result         = 1 - result_prec_score
                        result_roc_auc = f1_score(y_valid, predicted_valid_set, average='weighted')
                        
                        valid_result_folds.append(result)
                        valid_result_auc_folds.append(result_roc_auc)
                        
                    else:
                        result = mean_absolute_error(y_valid, predicted_valid_set)
                        print ("MAE: ", result)
                        result_rmse = sqrt(mean_squared_error(y_valid, predicted_valid_set))
                        print ("RMSE: ", result_rmse)
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
        self.fi_total['Importance_AVG']      = np.round(self.fi_total[fi_cols].sum(axis=1)/fold_all, decimals=10)  
        self.fi_total['Importance_AVG_ABS']  = self.fi_total['Importance_AVG'].abs()
        
        imp_avg_sum = self.fi_total['Importance_AVG_ABS'].sum(axis=0)
        if imp_avg_sum != 0:
            self.fi_total['Importance_AVG_ABS_perc'] = round(100 * self.fi_total['Importance_AVG_ABS'] / imp_avg_sum, 2)
        else:
            self.fi_total['Importance_AVG_ABS_perc'] = 0

        print ('\nFEATURE Importance Overall:')
        self.print_html( self.fi_total[['Feature','Importance_AVG','Importance_AVG_ABS','Importance_AVG_ABS_perc']].sort_values(by=['Importance_AVG_ABS'], ascending=False), max_rows=200, max_cols=4 )
           
        #print ('\nFEATURE Importance SHAP last validation:')
        #shap.initjs()
        #shap.summary_plot(valid_set_shap_values, df_valid_x[0:self.shap_data_limit])
        
        # save indexes used for splits
        self.dicts_agent['train_sub_sets_ix'] = train_sub_sets_ix
        self.dicts_agent['test_sub_sets_ix']  = test_sub_sets_ix

        self.dicts_agent['df_filter_column_mc'] = df_filter_column_mc
        self.dicts_agent['df_filter_column']    = df_filter_column
        
        # save performance summaries across all validation folds
        self.dicts_agent['fi_total']                               = self.fi_total
        #self.dicts_agent['fi_valid_shap']                          = valid_set_shap_values
        #self.dicts_agent['fi_valid_x']                             = df_valid_x[0:self.shap_data_limit]
        
        #############################################################
        #                   OUTPUT
        #############################################################
        fi_total_dict = dict(zip(self.fi_total['Feature'],self.fi_total['Importance_AVG_ABS_perc']))
        print ("#feature_importance="+json.dumps(fi_total_dict))
        
        if mode==1:  
            if self.params['random_folds'] == False:          
                df_filter_column[self.output_column] = float('nan')
                df_filter_column.loc[train_sets_ix[valid_fold], self.output_column] = prediction
                df_filter_column.loc[remainder_set_indexes,     self.output_column] = predicted_test_set

                if self.use_validation_set:
                    df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column] = predicted_valid_set
            else:
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
                
                # save models to files
                self.dicts_agent['models_saved'] = len(predictors)
                for fold in range(0, len(predictors)):
                    self.model_save(predictors[fold], workdir + self.output_column + "_fold" + str(fold) + ".model")
       
                if self.models_apply_on_all_data:
                    print (str(datetime.now())," --- Applying all models to all data --- ")
                    pred = np.zeros(len(df_all))
                    if self.params['algo']['objective'] == self.objective_multiclass:
                        # create a list of lists depending on number of classes used for training 
                        # as each prediction is a list of values against each class
                        pred = [np.zeros(self.params['algo']['num_class']) for i in range(len(df_all))]
                    
                    # apply model from each fold created during training and sum their predictions              
                    for fold in range(0, len(predictors)):
                        pred += self.model_predict(predictors[fold], df_all.drop(self.target_col, axis=1))

                    if self.params['algo']['objective'] == self.objective_multiclass:
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

            # save dictionary of all auxiliry data and params into file
            sfile = bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'w')
            pickle.dump(self.dicts_agent, sfile) 
            sfile.close()                 
            
            print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(original_row_count))
            
            print ("b_fitness=" +str(round(1-self.list_mean(weighted_auc_folds)*self.list_mean(valid_result_auc_folds),4)))
            print ("b_result_1="+str(self.list_mean(weighted_result_folds)))
            print ("b_result_2="+str(self.list_mean(weighted_auc_folds)))
            print ("b_result_3="+str(self.list_mean(valid_result_folds)))
            print ("b_result_4="+str(self.list_mean(valid_result_auc_folds)))
        else:
            print ("fitness="     +str(round(1-self.list_mean(weighted_auc_folds)*self.list_mean(valid_result_auc_folds),4)))  # main fitness metric
            print ("out_result_1="+str(self.list_mean(weighted_result_folds)))                                        # Log Loss in train/test CV
            print ("out_result_2="+str(self.list_mean(weighted_auc_folds)))                                           # ROC AUC in train/test CV
            print ("out_result_3="+str(self.list_mean(valid_result_folds)))                                           # main fitness on Validation
            print ("out_result_4="+str(self.list_mean(valid_result_auc_folds)))                                       # ROC AUC on Validation


ev_agent_{id} = cls_ev_agent_{id}()
