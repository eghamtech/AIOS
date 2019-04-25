#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=200
#key=fields_to_use;  type=random_int;  from=100;  to=200;  step=1
#key=field_ev_prefix;  type=random_from_set;  set=ev_field_cbst_
#key=nfolds;  type=random_int;  from=3;  to=3;  step=1
#key=random_folds;  type=random_from_set;  set=True
#key=random_folds_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=use_validation_set;  type=random_from_set;  set=True
#key=random_valid;  type=random_from_set;  set=True
#key=random_valid_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=random_valid_folds;  type=random_int;  from=10;  to=10;  step=1
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
#key=objective_multiclass;  type=random_from_set;  set='MultiClassOneVsAll'
#key=objective_regression;  type=random_from_set;  set='RMSE'
#key=bootstrap_type;  type=random_from_set;  set='Bayesian','Bernoulli','No'
#key=learning_rate;  type=random_float;  from=0.01;  to=0.09;  step=0.005
#key=metric_period;  type=random_int;  from=1;  to=5;  step=1
#key=bagging_temperature;  type=random_int;  from=1;  to=50;  step=1
#key=sampling_frequency;  type=random_from_set;  set='PerTree','PerTreeLevel'
#key=sampling_unit;  type=random_from_set;  set='Object','Group'
#key=random_strength;  type=random_float;  from=0.5;  to=1.5;  step=0.05
#key=l2_leaf_reg;  type=random_float;  from=0.5;  to=10;  step=0.05
#key=use_best_model;  type=random_from_set;  set=True,False
#key=depth;  type=random_int;  from=1;  to=16;  step=1
#key=grow_policy;  type=random_from_set;  set='SymmetricTree'
#key=min_data_in_leaf;  type=random_int;  from=1;  to=10;  step=1
#key=max_leaves;  type=random_int;  from=10;  to=64;  step=1
#key=iterations;  type=random_int;  from=100;  to=1500;  step=50
#key=has_time;  type=random_from_set;  set=True,False
#key=rsm;  type=random_float;  from=0.1;  to=1;  step=0.05
#key=nan_mode;  type=random_from_set;  set='Min','Max'
#key=fold_permutation_block;  type=random_int;  from=1;  to=50;  step=1
#key=one_hot_max_size;  type=random_int;  from=2;  to=50;  step=1
#key=leaf_estimation_method;  type=random_from_set;  set='Newton','Gradient'
#key=leaf_estimation_backtracking;  type=random_from_set;  set='AnyImprovement','No'
#key=fold_len_multiplier;  type=random_int;  from=1;  to=10;  step=1
#key=approx_on_full_history;  type=random_from_set;  set=True,False
#key=border_count;  type=random_int;  from=10;  to=255;  step=1
#key=feature_border_type;  type=random_from_set;  set='GreedyLogSum','MinEntropy','MaxLogSum','UniformAndQuantiles','Uniform','Median'
#key=od_type;  type=random_from_set;  set='Iter'
#key=od_wait;  type=random_int;  from=10;  to=20;  step=1
#key=verbose;  type=random_from_set;  set=0
#key=binary_balancing;  type=random_from_set;  set=False
#key=binary_balancing_0;  type=random_float;  from=0.1;  to=1;  step=0.02
#key=binary_balancing_1;  type=random_float;  from=0.1;  to=1;  step=0.02
#key=start_fold;  type=random_from_set;  set=0
#key=nthread;  type=random_int;  from=4;  to=4;  step=1
#key=use_float32_dtype; type=random_from_set;  set=True
#key=min_perf_criteria;  type=random_float;  from=0.6;  to=0.6;  step=0.1
#key=print_to_html; type=random_from_set;  set=True
#key=print_tables; type=random_from_set;  set=False
#end_of_genes_definitions

# AICHOO OS Evolving Agent
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

class cls_ev_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import random as rn
    import dateutil
    import calendar
    import os.path, bz2, pickle
    import pandas as pd
    import math
    import catboost as cbst

    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "field_ev_prefix" with unique instance ID
    # and filename to save new field data
    field_ev_prefix = "{field_ev_prefix}"
    output_column   = field_ev_prefix + str(result_id)
    output_filename = output_column + ".csv"

    # obtain random field (same for all instances within the evolution) which will be the prediction target for this instance/evolution
    target_definition = "{field_to_predict}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    target_col  = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]

    # obtain random selection of fields; number of fields to be selected specified in (data):length gene for this instance
    data_defs     = {data}
    fields_to_use = {fields_to_use}
    start_fold    = {start_fold}
    nfolds        = {nfolds}
    num_threads   = {nthread}
    rn_seed_init  = {random_seed_init}

    params        = {}             # ML algo parameters
    dicts_agent   = {}             # various dictionary to be saved as part of model

    # if filter columns are specified then training and validation sets will be selected based on filter criteria
    # based on filter criteria training + validation sets will not necessarily constitute all data, the remainder will be called "test set"
    filter_column   = "{filter_column}"
    filter_column_2 = "{filter_column_2}"

    # fields matching the specified string will not be used in the model
    ignore_columns_containing  = "{ignore_columns_containing}"
    # include only fields matching string e.g., only properly scaled columns should be used with MLP
    include_columns_containing = "{include_columns_containing}"

    objective_multiclass = {objective_multiclass}
    objective_regression = {objective_regression}

    print_tables  = {print_tables}
    print_to_html = {print_to_html}

    use_validation_set = {use_validation_set}
    use_float32_dtype  = {use_float32_dtype}
    min_perf_criteria  = {min_perf_criteria}

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

    #import os
    #os.environ['PYTHONHASHSEED'] = rn_seed_init

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(rn_seed_init)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(rn_seed_init)

    def is_set(self, s):
        return len(s)>0 and s!="0"

    def is_use_column(self, s):
        # AIOS Kernel now selects columns using agent parameters
        # so no need to filter inside the agent

        # determine whether given column should be used or ignored
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
            print (df.to_html(max_rows=max_rows, max_cols=max_cols))
        elif jup_notebook:
            display(df)
        else:
            print (df)

    def my_log_loss(self, a, b):
        eps  = 1e-9
        sum1 = 0.0
        for k in range(0, len(a)):
            bx = min(max(b[k],eps), 1-eps)
            sum1 += 1.0 * a[k] * self.math.log(bx) + 1.0 * (1 - a[k]) * self.math.log(1 - bx)
        return -sum1/len(a)

    def list_mean(self, lst, precision=4):
        return self.np.round(sum(lst)/float(len(lst)), decimals=precision)

    def prc_auc(self, train_y, pred):
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc

        precision, recall, thresholds = precision_recall_curve(train_y, pred)
        prc_auc = auc(recall, precision)

        return prc_auc


    def model_env_init(self):
        return None

    def model_params(self):
        self.params['iterations']                   = {iterations}
        self.params['learning_rate']                = {learning_rate}
        self.params['bootstrap_type']               = [bootstrap_type]
        self.params['metric_period']                = [metric_period]

        self.params['bagging_temperature']          = {bagging_temperature}
        self.params['sampling_frequency']           = {sampling_frequency}
        self.params['sampling_unit']                = {sampling_unit}
        self.params['random_strength']              = {random_strength}

        self.params['l2_leaf_reg']                  = {l2_leaf_reg}
        self.params['random_seed']                  = self.rn_seed_init


        self.params['use_best_model']               = {use_best_model}
        self.params['depth']                        = {depth}
        self.params['grow_policy']                  = {grow_policy}
        self.params['min_data_in_leaf']             = {min_data_in_leaf}
        self.params['max_leaves']                   = {max_leaves}
        self.params['one_hot_max_size']             = {one_hot_max_size}
        self.params['has_time']                     = {has_time}
        self.params['rsm']                          = {rsm}
        self.params['nan_mode']                     = {nan_mode}
        self.params['fold_permutation_block']       = {fold_permutation_block}
        self.params['leaf_estimation_method']       = {leaf_estimation_method}
        # self.params['leaf_estimation_iterations'] = None
        self.params['leaf_estimation_backtracking'] = {leaf_estimation_backtracking}
        self.params['fold_len_multiplier']          = {fold_len_multiplier}
        self.params['approx_on_full_history']       = {approx_on_full_history}
        # self.params['class_weights']            = [0.5, 1]
        self.params['border_count']                 = {border_count}
        self.params['feature_border_type']          = {feature_border_type}

        self.params['od_type']                      = {od_type}
        self.params['od_wait']                      = {od_wait}
        # self.params['od_pval']                    = 0
        self.params['thread_count']                 = {nthread}
        self.params['verbose']                      = {verbose}

        if self.is_binary:
            print ("detected binary target: use AUC/LOGLOSS and Binary Cross Entropy loss evaluation")
            self.params['objective']                     = 'CrossEntropy'
            self.params['eval_metric']                   = 'AUC'
            self.params['num_class']                     = 1
            # self.params['loss_function']               = 'crossentropy'
            # self.params['metric']                      = [self.tf_roc_auc, self.tf_prc_auc]                                 # if using custom metric function cannot save in params as pickle will fail
            # self.metric                                = [self.tf_roc_auc, self.tf_prc_auc]                                 # in such case use local class variable for metric
            # self.params['early_stop_metric']           = 'val_tf_prc_auc'
            # self.params['early_stop_metric_direction'] = 'max'
        elif self.is_set(self.objective_multiclass):
            print ("detected multi-class target: use Multi-LogLoss/Error; " + str(len(self.target_classes)) + " classes")
            self.params['objective']                     = self.objective_multiclass
            self.params['eval_metric']                   = 'MultiClassOneVsAll'
            self.params['num_class']                     = int(max(self.target_classes) + 1)  # requires all int numbers from 0 to max to be classes
            # self.params['loss_function']               = 'MultiClassOneVsAll'
            # self.params['metric']                      = ['accuracy']
            # self.metric                                = ['accuracy']
            # self.params['early_stop_metric']           = 'val_loss'
            # self.params['early_stop_metric_direction'] = 'auto'
        else:
            print ("detected regression target: use RMSE/MAE")
            self.params['objective']                     = self.objective_regression
            self.params['eval_metric']                   = 'RMSE'
            # self.params['loss_function']               = 'RMSE'
            # self.params['metric']                      = ['mean_squared_error']
            # self.metric                                = ['mean_squared_error']
            # self.params['early_stop_metric']           = 'val_loss'
            # self.params['early_stop_metric_direction'] = 'auto'
            self.params['num_class']                     = 1
            # params['metric']             = ['rmse', 'mae']

    def model_init(self):
        ml_model = self.cbst.CatBoost(self.params)
        return ml_model

    def model_predict(self, predictor, xt):
        try:
            cat_features_ind = self.np.where(self.np.logical_and(xt.dtypes != self.np.float32, xt.dtypes != self.np.float64))[0]

            xt   = self.cbst.Pool(xt, cat_features=cat_features_ind, feature_names=list(xt.columns))
            pred = predictor.predict(xt, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0,
                                     thread_count=self.params['thread_count'], verbose=self.params['verbose'])

        except Exception as e:
            print ('CatBoost Predict error: ', e)
            pred = 0

        return pred

    def model_save(self, predictor, file_path):
        predictor.save_model(file_path)

    def model_load(self, file_path):
        predictor = self.cbst.CatBoost()
        predictor.load_model(file_path)

        return predictor

    def model_feature_importance(self, predictor, n_top_features=25, col_idx=0, importance_type='gain', feat_names=[], print_table=True, to_html=True):
        importance = predictor.get_feature_importance()
        features   = predictor.feature_names_
        # join field names and their importance values
        col_name = 'Importance_' + str(col_idx)
        fi = self.pd.DataFrame({'Feature': features, col_name: importance})
        fi[col_name] = fi[col_name].round(4)

        if col_idx == 1:
            self.fi_total = fi
        else:
            self.fi_total = self.pd.merge(self.fi_total, fi, how='outer', on='Feature', sort=False)

        if print_table:
            print ()
            self.print_html(fi.sort_values(by=[col_name], ascending=False), max_rows=n_top_features * 2, max_cols=2)

    def model_train(self, ml_model, x_train, y_train, x_test, y_test, current_fold):
        cat_features_ind = self.np.where(self.np.logical_and(x_train.dtypes != self.np.float32, x_train.dtypes != self.np.float64))[0]

        x_train = self.cbst.Pool(x_train, label=y_train, cat_features=cat_features_ind, feature_names=list(x_train.columns))
        x_test  = self.cbst.Pool(x_test,  label=y_test,  cat_features=cat_features_ind, feature_names=list(x_test.columns))

        ml_model = ml_model.fit(x_train, eval_set=x_test)

        self.model_feature_importance(ml_model, n_top_features=25, col_idx=current_fold, importance_type='gain', print_table=self.print_tables, to_html=self.print_to_html)

        return ml_model


    def load_columns(self, map_dict=True):
        from datetime import datetime
        # start from loading the target field
        df_all = self.pd.read_csv(workdir + self.target_file, usecols=[self.target_col])[[self.target_col]]

        columns_new = [self.target_col]
        columns     = [self.target_col]
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        print (str(datetime.now()), " start loading data")
        block_progress = 0
        block          = int(self.fields_to_use / 20)

        for i in range(0, self.fields_to_use):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if self.is_use_column(col_name):
                df_col = self.pd.read_csv(workdir + file_name, usecols=[col_name])[[col_name]]            # read column from csv file

                # if column has associated dictionary csv then it's a text column, replace column with actual text
                dict_file_name = workdir + 'dict_' + col_name + '.csv'
                if self.os.path.isfile(dict_file_name) and map_dict:
                    dict1 = self.pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()  # load dictionary
                    df_col[col_name] = df_col[col_name].map(dict1)                                                         # map and replace
                    self.dicts_agent[col_name] = dict1                                                                     # save in dictionary of dictionaries to be saved with model files

                    #df_col[col_name] = df_col[col_name].astype(str).apply(self.clean_text)
                else:
                    if df_col[col_name].dtype == self.np.float64 and self.use_float32_dtype:              # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(self.np.float32)

                df_all = df_all.merge(df_col, left_index=True, right_index=True)                          # add column to the overall dataframe

                block_progress += 1
                if block_progress >= block:
                    block_progress = 0
                    print (str(datetime.now()), " data loaded: ", round((i + 1) / self.fields_to_use * 100, 0), "%")

                # some columns may appear multiple times in data_defs as inhereted from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                columns.append(col_name)
                ncol_count = columns.count(col_name)
                if ncol_count == 1:
                    columns_new.append(col_name)
                else:
                    columns_new.append(col_name + "_v" + str(ncol_count))

        # rename columns in df to unique names
        df_all.columns = columns_new
        print (str(datetime.now()), " data loaded", len(df_all), "rows; ", len(df_all.columns), "columns")
        return df_all


    def __init__(self):
        from datetime import datetime
        # remove the target field for this instance from the data used for training
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)

        self.model_env_init()

        if self.os.path.isfile(workdir + self.output_column + '_dicts.model'):
            rfile = self.bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'r')
            self.dicts_agent = self.pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.output_column + ' dictionaries model loaded')

            # if saved model for the target field already exists then load it from filesystem
            self.predictors = []
            if self.dicts_agent['params']['random_folds'] == False:
                from_fold = self.start_fold
                to_fold   = self.nfolds
            else:
                from_fold = 0
                to_fold   = 3  # use fixed 3 saved models to make any prediction

            for fold in range(from_fold, to_fold):
                if self.os.path.isfile(workdir + self.output_column + "_fold" + str(fold) + ".model"):
                    predictor_stored = self.model_load(workdir + self.output_column + "_fold" + str(fold) + ".model")
                    self.predictors.append(predictor_stored)
                    print (str(datetime.now()), self.output_column + ' fold ' + str(fold) + ' predictor model loaded')

        # obtain columns definitions to filter data set by
        if self.is_set(self.filter_column):
            self.filter_filename = self.filter_column.split("|")[1]
            self.filter_column   = self.filter_column.split("|")[0]

        if self.is_set(self.filter_column_2):
            self.filter_filename_2 = self.filter_column_2.split("|")[1]
            self.filter_column_2   = self.filter_column_2.split("|")[0]


    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        # df_add shouldn't contain columns with text values - only numeric
        columns_new = []
        columns     = []
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        cols_count = 0
        for i in range(0,self.fields_to_use):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if self.is_use_column(col_name):
                # assemble dataframe column by column
                df_col = df_add[[col_name]]

                if df_col[col_name].dtype == self.np.float64 and self.use_float32_dtype:  # downcast to save memory if needed
                    df_col[col_name] = df_col[col_name].astype(self.np.float32)

                if i == 0:
                    df = df_col[[col_name]]
                else:
                    df = df.merge(df_col[[col_name]], left_index=True, right_index=True)

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

        # predict new data set in df applying model for each fold used for training
        pred = self.np.zeros(len(df))
        if self.dicts_agent['params']['objective'] == self.objective_multiclass:
            # create a list of lists depending on number of classes used for training
            # as each prediction is a list of values against each class
            pred = [self.np.zeros(self.dicts_agent['params']['num_class']) for i in range(len(df))]

        if self.dicts_agent['params']['random_folds'] == False:
            for fold in range(self.start_fold, self.nfolds):
                pred += self.model_predict(predictors[fold - self.start_fold], df)

            if self.dicts_agent['params']['objective'] == self.objective_multiclass:
                # select class with largest total value in case of multiclass
                pred = self.np.argmax(pred, axis=1)
            else:
                # average prediction over all folds in case of binary or regression
                pred = pred / (self.nfolds - self.start_fold)
        else:
            for fold in range(0, len(self.predictors)):
                pred += self.model_predict(predictors[fold], df)

            if self.dicts_agent['params']['objective'] == self.objective_multiclass:
                # select class with largest total value in case of multiclass
                pred = self.np.argmax(pred, axis=1)
            else:
                # average prediction over all folds in case of binary or regression
                pred = pred / len(self.predictors)

        df_add[self.output_column] = pred


    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, log_loss
        from sklearn.metrics import confusion_matrix, f1_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import StratifiedShuffleSplit
        from math import sqrt
        from datetime import datetime
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

        # obtain indexes for train and remainder sets
        # load target column as it may be needed for filtering and removing NaN targets from training
        df_filter_column       = self.pd.read_csv(workdir + self.target_file, usecols=[self.target_col])
        filter_condition_train = df_filter_column[self.target_col].notnull()

        # applying specified filters
        if self.is_set(self.filter_column):
            # load columns to filter by
            df_t = self.pd.read_csv(workdir + self.filter_filename, usecols=[self.filter_column])
            df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)

            filter_condition_train = self.np.logical_and( filter_condition_train,
                                        self.np.logical_and( df_filter_column[self.filter_column] >= {train_set_from},
                                                             df_filter_column[self.filter_column] <  {train_set_to} ) )

            # two filter columns specified
            if self.is_set(self.filter_column_2):
                df_t = self.pd.read_csv(workdir + self.filter_filename_2, usecols=[self.filter_column_2])
                df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)

                condition2 = self.np.logical_and(df_filter_column[self.filter_column_2] >= {train_set_from_2},
                                                 df_filter_column[self.filter_column_2] <  {train_set_to_2} )
                filter_condition_train = self.np.logical_and(filter_condition_train, condition2)

        train_filtered_indexes = df_filter_column[filter_condition_train].index.tolist()
        remainder_set_indexes  = df_filter_column[self.np.logical_not(filter_condition_train)].index.tolist()  # remainder which is not in train

        # initialise prediction column for entire data set as it will be aggregate prediction from multiple folds
        df_filter_column[self.output_column + '_folds_pred']       = 0
        df_filter_column[self.output_column + '_folds_pred_count'] = 0  # number of predictions for each record as different folds will predict different records, so each record may have unique number of predictions

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
        df_filter_column_mc = self.pd.DataFrame([self.np.zeros(self.params['num_class']) for i in range(len(df_filter_column))])

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

        fold_all = 0
        # repeat cross-validation multiple times with different validation set each time
        # applies only in case when params['random_valid'] == True
        for valid_fold in range(0, self.params['random_valid_folds']):
            print ()
            print (str(datetime.now()), " ----- VALID FOLD: ", valid_fold)
            # obtain indexes for validation set if required
            # applying specified filters
            if self.use_validation_set:
                # assemble condition for filtering validation set
                filter_condition_valid = df_filter_column[self.target_col].notnull()

                if self.is_set(self.filter_column):
                    filter_condition_valid = self.np.logical_and(filter_condition_valid,
                                                                 self.np.logical_and( df_filter_column[self.filter_column] >= {valid_set_from},
                                                                                      df_filter_column[self.filter_column] <  {valid_set_to} ) )
                    # two filter columns specified
                    if self.is_set(self.filter_column_2):
                        condition2 = self.np.logical_and( df_filter_column[self.filter_column_2] >= {valid_set_from_2},
                                                          df_filter_column[self.filter_column_2] <  {valid_set_to_2} )
                        filter_condition_valid = self.np.logical_and(filter_condition_valid, condition2)

                if self.params['random_valid'] == False:
                    # select validation based on fixed filter - may intersect with test or remainder set
                    train_sets_ix.append(train_filtered_indexes)
                    valid_sets_ix.append(df_filter_column[filter_condition_valid].index.tolist())
                else:
                    # apply stratified random selection to previously filtered train set
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=self.params['random_valid_size'])
                    y   = df_filter_column[df_filter_column.index.isin(train_filtered_indexes)][[self.target_col]]
                    iy  = y.reset_index(level=0)                                              # create copy, save existing index in 'index' column and reset index
                    y.reset_index(drop=True, inplace=True)                                    # reset index because StratifiedShuffleSplit will reset index anyway

                    for train_ix, valid_ix in sss.split(self.np.zeros(len(y)), y):
                        train_sets_ix.append( iy[iy.index.isin(train_ix)]['index'].tolist())  # obtain original indexes from saved copy of labels with original indexes
                        valid_sets_ix.append( iy[iy.index.isin(valid_ix)]['index'].tolist())  # can't use train_ix, valid_ix directly because they refer to new index reset during shuffling
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
            df      = df_all.copy()
            # use previously calculated indexes to select train, validation and remainder sets
            df_test = df[df.index.isin(remainder_set_indexes)]

            if self.use_validation_set:
                df_valid = df[df.index.isin(valid_sets_ix[valid_fold])]
                # initialise prediction column for validation as it will be aggregate prediction from multiple folds
                predicted_valid_set = self.np.zeros(len(df_valid))
                # Multi-class case: initialise prediction list of lists depending on number of classes
                # as each prediction is a list of values against each class
                if self.params['objective'] == self.objective_multiclass:
                    predicted_valid_set = [self.np.zeros(self.params['num_class']) for i in range(len(df_valid))]

            df = df[df.index.isin(train_sets_ix[valid_fold])]

            # initialise prediction column for main train set as it will be aggregate prediction from multiple folds
            prediction         = self.np.zeros(len(df))
            # initialise prediction column for remainder set as it will be aggregate prediction from multiple folds
            predicted_test_set = self.np.zeros(len(df_test))
            # Multi-class case: initialise prediction list of lists depending on number of classes
            # as each prediction is a list of values against each class
            if self.params['objective'] == self.objective_multiclass:
                prediction         = [self.np.zeros(self.params['num_class']) for i in range(len(df))]
                predicted_test_set = [self.np.zeros(self.params['num_class']) for i in range(len(df_test))]

            #############################################################
            #                   MAIN LOOP
            #############################################################

            weighted_result = 0
            weighted_auc    = 0
            count_records_notnull = 0

            if self.params['random_folds'] == False:
                # divide training data into nfolds of size block
                block = int(len(df) / self.nfolds)
                # select folds sequentially in existing index order
                for fold in range(self.start_fold, self.nfolds):
                    print ()
                    print (str(datetime.now()), " Train/Test FOLD: ", fold)
                    range_start = fold*block
                    range_end   = (fold+1)*block
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

                    y_train = x_train[self.target_col]          # separate training fields and the target
                    x_train = x_train.drop(self.target_col, 1)

                    y_test = x_test[self.target_col]
                    x_test = x_test.drop(self.target_col, 1)

                    predictor = self.model_init()
                    predictor = self.model_train(predictor, x_train, y_train, x_test, y_test, fold-self.start_fold+1)
                    pred      = self.model_predict(predictor, x_test)

                    if mode==1:
                        self.model_save(predictor, workdir + self.output_column + "_fold" + str(fold) + ".model")

                    if self.is_binary:
                        result = my_log_loss(y_test, pred)
                        # show various metrics as per
                        # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                        result_roc_auc = roc_auc_score(y_test, pred)
                        result_prc_auc = self.prc_auc(y_test, pred)
                        print ("ROC AUC score: ", result_roc_auc)
                        print ("PRC AUC score: ", result_prc_auc)

                        if self.print_tables:
                            result_cm = confusion_matrix(y_test, (pred > 0.5))  # assume 0.5 probability threshold
                            result_cr = classification_report(y_test, (pred > 0.5))
                            print ("Confusion Matrix:\n", result_cm)
                            print ("Classification Report:\n", result_cr)
                    elif self.is_set(self.objective_multiclass):
                        pred_classes = self.np.argmax(pred, axis=1)
                        result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                        result_acc_score  = accuracy_score(y_test, pred_classes)
                        result_cm = confusion_matrix(y_test, pred_classes)
                        result_cr = classification_report(y_test, pred_classes)
                        if self.print_tables:
                            print ("Precision score: ", result_prec_score)
                            print ("Accuracy score: ", result_acc_score)
                            print ("Confusion Matrix:\n", result_cm)
                            print ("Classification Report:\n", result_cr)
                        result = predictor.best_score['valid_0']['multi_logloss']
                        result_roc_auc = f1_score(y_test, pred_classes, average='weighted')
                    else:
                        result = sum(abs(y_test - pred)) / len(y_test)
                        # result = sqrt(mean_squared_error(y_test, pred))

                    print ("result: ", result)

                    if result_roc_auc < self.min_perf_criteria:
                        print ("Minimum performance criteria: " + str(self.min_perf_criteria) + " not met! result_roc_auc: " + str(result_roc_auc))
                        return

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
                    count_records_notnull += len(pred)

                    # predict all examples in the original test set which may include erroneous examples previously removed
                    pred_all_test = self.model_predict(predictor, x_test_orig.drop(self.target_col, axis=1))

                    if self.params['objective'] == self.objective_multiclass:
                        prediction[range_start:range_end] = self.np.argmax(pred_all_test, axis=1)
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
                sss = StratifiedShuffleSplit(n_splits=self.nfolds, test_size=self.params['random_folds_size'])
                y   = df[[self.target_col]]
                iy  = y.reset_index(level=0)            # create copy, save existing index in 'index' column and reset index
                y.reset_index(drop=True, inplace=True)  # reset index because StratifiedShuffleSplit will reset index anyway

                predictors = []
                for train_ix, test_ix in sss.split(self.np.zeros(len(y)), y):
                    fold_all += 1
                    print ()
                    print (str(datetime.now()), " Train/Test FOLD: ", fold_all)

                    train_ix_orig = iy[iy.index.isin(train_ix)]['index'].tolist()  # obtain original indexes from saved copy of labels with original indexes
                    test_ix_orig  = iy[iy.index.isin(test_ix)]['index'].tolist()   # can't use train_ix, test_ix directly because they refer to new index reset during shuffling

                    # ------ balance train set -----------------------------------------------------------------------------------------------------
                    if self.params['binary_balancing']:
                        bal_y = df[[self.target_col]]
                        # undersample both binary label samples by fixed per label percentage
                        bal_cond = self.np.logical_and(bal_y.index.isin(train_ix_orig), bal_y[self.target_col] == 0)
                        train_ix_orig_balanced_0 = bal_y[bal_cond].index.tolist()
                        train_balanced_size_0    = int(len(train_ix_orig_balanced_0) * self.params['binary_balancing_0'])
                        train_ix_orig_balanced_0 = self.np.random.choice(train_ix_orig_balanced_0, train_balanced_size_0, replace=False).tolist()

                        bal_cond = self.np.logical_and(bal_y.index.isin(train_ix_orig), bal_y[self.target_col] == 1)
                        train_ix_orig_balanced_1 = bal_y[bal_cond].index.tolist()
                        train_balanced_size_1    = int(len(train_ix_orig_balanced_1) * self.params['binary_balancing_1'])
                        train_ix_orig_balanced_1 = self.np.random.choice(train_ix_orig_balanced_1, train_balanced_size_1, replace=False).tolist()

                        train_ix_orig = train_ix_orig_balanced_0 + train_ix_orig_balanced_1
                    # ------------------------------------------------------------------------------------------------------------------------------

                    train_sub_sets_ix.append(train_ix_orig)  # save indexes in the overall list for all folds
                    test_sub_sets_ix.append(test_ix_orig)

                    x_test  = df[df.index.isin(test_ix_orig)]
                    x_train = df[df.index.isin(train_ix_orig)]

                    print ("x_test  rows count: " + str(len(x_test)))
                    print ("x_train rows count: " + str(len(x_train)))

                    y_train = x_train[self.target_col]  # separate training fields and the target
                    x_train = x_train.drop(self.target_col, 1)

                    y_test = x_test[self.target_col]
                    x_test = x_test.drop(self.target_col, 1)

                    predictor = self.model_init()
                    predictor = self.model_train(predictor, x_train, y_train, x_test, y_test, fold_all)
                    pred      = self.model_predict(predictor, x_test)

                    try:
                        if self.is_binary:
                            result = log_loss(y_test, pred)
                            # show various metrics as per
                            # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                            result_roc_auc = roc_auc_score(y_test, pred)
                            result_prc_auc = self.prc_auc(y_test, pred)
                            print ("ROC AUC score: ", result_roc_auc)
                            print ("PRC AUC score: ", result_prc_auc)

                            if self.print_tables:
                                result_cm = confusion_matrix(y_test, self.np.asarray(pred) > 0.5)  # assume 0.5 probability threshold
                                result_cr = classification_report(y_test, self.np.asarray(pred) > 0.5)
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred_count'] += 1

                        elif self.params['objective'] == self.objective_multiclass:
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

                            result         = log_loss(y_test, pred)
                            result_roc_auc = f1_score(y_test, pred_classes, average='weighted')

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_pred = self.np.array(df_filter_column_mc.loc[test_ix_orig])  # get array of previous folds test records predictions
                            df_pred += pred
                            df_filter_column_mc.loc[test_ix_orig] = df_pred                 # temp df holding multi-class prediction
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred_count'] += 1

                        else:
                            result = sum(abs(y_test - pred)) / len(y_test)
                            # result = sqrt(mean_squared_error(y_test, pred))

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred_count'] += 1
                    except Exception as e:
                        print (e)
                        result = 999999
                        result_roc_auc = 0

                    print ("result: ", result)

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
                    count_records_notnull += len(pred)

                    if result_roc_auc < self.min_perf_criteria:
                        print ("Minimum performance criteria: " + str(self.min_perf_criteria) + " not met! result_roc_auc: " + str(result_roc_auc))
                        return

                    predictors.append([predictor, result, result_roc_auc])
                    predictors_all.append([predictor, result, result_roc_auc])  # add predictors to global list across all validation folds
                #-------------- end of train test CV loop ---------------------------------------------------------------------------------------------

                predictors = self.pd.DataFrame(predictors, columns=['predictor', 'result', 'result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
                print ('\nFolds Performance Overall:')
                self.print_html(predictors, max_rows=50, max_cols=5)

                # select 3 predictors (best, worst and average) to be used for predicting all validation and remaining samples
                predictors['result_roc_auc_mean']      = predictors['result_roc_auc'].mean()
                predictors['result_roc_auc_mean_diff'] = abs(predictors['result_roc_auc'] - predictors['result_roc_auc_mean'])

                best_predictor_idx  = predictors['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors['result_roc_auc'].idxmin()
                avg_predictor_idx   = predictors['result_roc_auc_mean_diff'].idxmin()

                predictors = [ predictors['predictor'][worst_predictor_idx],
                               predictors['predictor'][avg_predictor_idx],
                               predictors['predictor'][best_predictor_idx] ]

                print('Selected predictor ids: ', [worst_predictor_idx, avg_predictor_idx, best_predictor_idx])

                #------------------ predict remaining and validation samples --------------------------------------------
                for fold in range(0, len(predictors)):
                    # predict remainder in the column output mode
                    if len(df_test) > 0 and mode == 1:
                        pred = self.model_predict(predictors[fold], df_test.drop(self.target_col, axis=1))
                        predicted_test_set += pred

                        if self.params['objective'] == self.objective_multiclass:
                            # assign predictions to corresponding test records only
                            df_pred = self.np.array(df_filter_column_mc.loc[remainder_set_indexes])      # get array of previous folds test records predictions
                            df_pred += pred
                            df_filter_column_mc.loc[remainder_set_indexes] = df_pred                     # temp df holding multi-class prediction
                            df_filter_column.loc[remainder_set_indexes, self.output_column + '_folds_pred_count'] += 1
                        else:
                            df_filter_column.loc[remainder_set_indexes, self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[remainder_set_indexes, self.output_column + '_folds_pred_count'] += 1

                    # predict validation set
                    if self.use_validation_set:
                        df_valid_x = df_valid.drop(self.target_col, axis=1)
                        pred = self.model_predict(predictors[fold], df_valid_x)
                        predicted_valid_set += pred

                        if self.params['objective'] == self.objective_multiclass:
                            # assign predictions to corresponding test records only
                            df_pred = self.np.array(df_filter_column_mc.loc[valid_sets_ix[valid_fold]])  # get array of previous folds test records predictions
                            df_pred += pred
                            df_filter_column_mc.loc[valid_sets_ix[valid_fold]] = df_pred                 # temp df holding multi-class prediction
                            df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column + '_folds_pred_count'] += 1
                        else:
                            df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column + '_folds_pred_count'] += 1

                        #if fold == 0:
                        #    valid_set_shap_values = shap.TreeExplainer(predictors[fold]).shap_values(df_valid_x)
                        #else:
                        #    valid_set_shap_values += shap.TreeExplainer(predictors[fold]).shap_values(df_valid_x)
                # ------------------ end of predicting remaining and validation samples ---------------------------------

                prediction = prediction / len(predictors)
                predicted_test_set  = predicted_test_set / len(predictors)
                predicted_valid_set = predicted_valid_set / len(predictors)

                df_filter_column[self.output_column + '_folds_pred_avg'] = df_filter_column[self.output_column + '_folds_pred'] / df_filter_column[self.output_column + '_folds_pred_count']
            #------------ end of train test CV method selection ---------------------------------------------------------

            weighted_result = weighted_result/count_records_notnull
            weighted_auc    = weighted_auc/count_records_notnull

            weighted_result_folds.append(weighted_result)
            weighted_auc_folds.append(weighted_auc)

            print ("\nweighted_result: ", weighted_result)
            print ("weighted_auc: ",      weighted_auc)

            # if multiclass convert list of lists into list of predicted labels
            if self.params['objective'] == self.objective_multiclass:
                predicted_valid_set = self.np.argmax(predicted_valid_set, axis=1)
                predicted_test_set  = self.np.argmax(predicted_test_set, axis=1)

            if self.use_validation_set:
                print()
                print ("*************  VALIDATION SET RESULTS  *****************")
                print ("Length of validation set:", len(predicted_valid_set))

                # validation set may have missing labels (NAN), for metrics calc find subset with proper labels
                df_valid['predicted_valid_set'] = predicted_valid_set
                df_valid = df_valid[df_valid[self.target_col].notnull()]
                #df_valid.reset_index(drop=True, inplace=True)
                y_valid             = self.np.array(df_valid[self.target_col])
                predicted_valid_set = self.np.array(df_valid['predicted_valid_set'])

                try:
                    if self.is_binary:
                        result = log_loss(y_valid, predicted_valid_set)
                        print ("LOGLOSS: ", result)
                        result_roc_auc = roc_auc_score(y_valid, predicted_valid_set)
                        print ("ROC AUC score: ", result_roc_auc)
                        result_prc_auc = self.prc_auc(y_valid, predicted_valid_set)
                        print ("PRC AUC score: ", result_prc_auc)

                        if self.print_tables:
                            result_cm = confusion_matrix(y_valid, (predicted_valid_set > 0.5))  # assume 0.5 probability threshold
                            print ("Confusion Matrix:\n",      result_cm)
                            result_cr = classification_report(y_valid, (predicted_valid_set > 0.5))
                            print ("Classification Report:\n", result_cr)

                        valid_result_folds.append(result)
                        valid_result_auc_folds.append(result_roc_auc)

                    elif self.params['objective'] == self.objective_multiclass:
                        result_prec_score = precision_score(y_valid, predicted_valid_set, average='weighted')
                        result_acc_score  = accuracy_score(y_valid, predicted_valid_set)
                        result_cm = confusion_matrix(y_valid, predicted_valid_set)
                        result_cr = classification_report(y_valid, predicted_valid_set)

                        if self.print_tables:
                            print ("Precision score: ",        result_prec_score)
                            print ("Accuracy score: ",         result_acc_score)
                            print ("Confusion Matrix:\n",      result_cm)
                            print ("Classification Report:\n", result_cr)

                        result = 1 - result_prec_score
                        result_roc_auc = f1_score(y_valid, predicted_valid_set, average='weighted')

                    else:
                        # result = sum(abs(y_valid-predicted_valid_set))/len(y_valid)
                        # print ("MAE: ", result)
                        result = sqrt(mean_squared_error(y_valid, predicted_valid_set))
                        valid_result_folds.append(result)
                        print ("Root Mean Squared Error: ", result)
                except Exception as e:
                    print (e)
                    return  # no point to carry on with more folds

                print ("\n************* END of VALIDATION SET RESULTS  ****************\n")
        #----------- end of validation sets loop --------------------------------------------------------------------

        print ('\nTrain/Valid Folds Predictor Performance Overall:')
        predictors_all = self.pd.DataFrame(predictors_all, columns=['predictor', 'result', 'result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
        self.print_html(predictors_all, max_rows=50, max_cols=5)

        # combine feature importance results from all folds into one table
        fi_cols = [col for col in self.fi_total.columns if 'Importance' in col]
        self.fi_total['Importance_AVG']      = self.np.round(self.fi_total[fi_cols].sum(axis=1) / fold_all, decimals=2)
        self.fi_total['Importance_AVG_perc'] = self.np.round(100 * self.fi_total['Importance_AVG'] / self.fi_total['Importance_AVG'].sum(axis=0), decimals=2)

        print ('\nFEATURE Importance Overall:')
        self.print_html( self.fi_total[['Feature', 'Importance_AVG', 'Importance_AVG_perc']].sort_values(by=['Importance_AVG'], ascending=False), max_rows=200, max_cols=4)

        # print ('\nFEATURE Importance SHAP last validation:')
        # shap.initjs()
        # shap.summary_plot(valid_set_shap_values, df_valid_x)

        # save indexes used for splits
        self.dicts_agent['train_sub_sets_ix'] = train_sub_sets_ix
        self.dicts_agent['test_sub_sets_ix']  = test_sub_sets_ix

        # save performance summaries across all validation folds
        self.dicts_agent['fi_total'] = self.fi_total
        #self.dicts_agent['fi_valid_shap'] = valid_set_shap_values
        #self.dicts_agent['fi_valid_x'] = df_valid_x

        #############################################################
        #                   OUTPUT
        #############################################################

        if mode == 1:
            # save dictionary of all auxiliry data and params into file
            sfile = self.bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'w')
            self.pickle.dump(self.dicts_agent, sfile)
            sfile.close()

            if self.params['random_folds'] == False:
                df_filter_column[self.output_column] = float('nan')
                df_filter_column.loc[train_sets_ix[valid_fold], self.output_column] = prediction
                df_filter_column.loc[remainder_set_indexes, self.output_column]     = predicted_test_set

                if self.use_validation_set:
                    df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column] = predicted_valid_set
            else:
                # select 3 models from all train/test/valid folds
                predictors_all['result_roc_auc_mean']      = predictors_all['result_roc_auc'].mean()
                predictors_all['result_roc_auc_mean_diff'] = abs(predictors_all['result_roc_auc'] - predictors_all['result_roc_auc_mean'])

                best_predictor_idx  = predictors_all['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors_all['result_roc_auc'].idxmin()
                avg_predictor_idx   = predictors_all['result_roc_auc_mean_diff'].idxmin()

                predictors = [ predictors_all['predictor'][worst_predictor_idx],
                               predictors_all['predictor'][avg_predictor_idx],
                               predictors_all['predictor'][best_predictor_idx] ]

                print('Selected predictor ids: ', [worst_predictor_idx, avg_predictor_idx, best_predictor_idx])

                for fold in range(0, len(predictors)):
                    self.model_save(predictors[fold], workdir + self.output_column + "_fold" + str(fold) + ".model")

                # if multiclass convert list of lists into list of predicted labels
                if self.params['objective'] == self.objective_multiclass:
                    df_filter_column[self.output_column + '_folds_pred'] = self.np.argmax(self.np.array(df_filter_column_mc), axis=1)
                    df_filter_column[self.output_column] = df_filter_column[self.output_column + '_folds_pred']
                    df_filter_column.loc[df_filter_column[self.output_column + '_folds_pred_count'] == 0, self.output_column] = float('nan')
                else:
                    df_filter_column[self.output_column] = df_filter_column[self.output_column + '_folds_pred'] / df_filter_column[self.output_column + '_folds_pred_count']

            df_filter_column[[self.output_column]].to_csv(workdir + self.output_filename)
            print ("#add_field:" + self.output_column + ",N," + self.output_filename + "," + str(original_row_count))

            print ("b_fitness="    + str(1 - self.list_mean(weighted_auc_folds) * self.list_mean(valid_result_auc_folds)))
            print ("b_result_1="   + str(self.list_mean(weighted_result_folds)))
            print ("b_result_2="   + str(self.list_mean(weighted_auc_folds)))
            print ("b_result_3="   + str(self.list_mean(valid_result_folds)))
            print ("b_result_4="   + str(self.list_mean(valid_result_auc_folds)))
        else:
            print ("fitness="      + str(1 - self.list_mean(weighted_auc_folds) * self.list_mean(valid_result_auc_folds)))  # main fitness metric
            print ("out_result_1=" + str(self.list_mean(weighted_result_folds)))                                            # Log Loss in train/test CV
            print ("out_result_2=" + str(self.list_mean(weighted_auc_folds)))                                               # ROC AUC in train/test CV
            print ("out_result_3=" + str(self.list_mean(valid_result_folds)))                                               # main fitness on Validation
            print ("out_result_4=" + str(self.list_mean(valid_result_auc_folds)))                                           # ROC AUC on Validation


ev_agent_{id} = cls_ev_agent_{id}()
