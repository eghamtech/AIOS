#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=13
#key=fields_to_use;  type=random_int;  from=13;  to=13;  step=1
#key=nfolds;  type=random_int;  from=10;  to=10;  step=1
#key=use_validation_set;  type=random_from_set;  set=False
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
#key=ignore_columns_containing;  type=random_from_set;  set=%ev_field%
#key=include_columns_containing;  type=random_from_set;  set=
#key=objective_multiclass;  type=random_from_set;  set='multiclass','multiclassova'
#key=objective_regression;  type=random_from_set;  set='regression_l1','regression_l2','huber','fair','poisson','quantile','mape','gamma','tweedie'
#key=boosting_type;  type=random_from_set;  set='gbdt','rf','dart','goss'
#key=learning_rate;  type=random_float;  from=0.001;  to=0.06;  step=0.001
#key=sub_feature;  type=random_float;  from=0.2;  to=1;  step=0.01
#key=bagging_fraction;  type=random_float;  from=0.2;  to=1;  step=0.01
#key=bagging_freq;  type=random_int;  from=10;  to=100;  step=1
#key=num_leaves;  type=random_int;  from=16;  to=4096;  step=1
#key=tree_learner;  type=random_from_set;  set='serial','feature','data','voting'
#key=min_data;  type=random_int;  from=100;  to=2000;  step=5
#key=feature_fraction_seed;  type=random_int;  from=1;  to=10;  step=1
#key=bagging_seed;  type=random_int;  from=1;  to=10;  step=1
#key=boost_from_average;  type=random_from_set;  set=True,False
#key=is_unbalance;  type=random_from_set;  set=True,False
#key=lambda_l1;  type=random_float;  from=0;  to=1;  step=0.01
#key=lambda_l2;  type=random_float;  from=0;  to=1;  step=0.01
#key=start_fold;  type=random_from_set;  set=0
#key=max_depth;  type=random_int;  from=-1;  to=10;  step=1
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

    import pandas as pd
    import lightgbm as lgb
    import numpy as np
    import math
    import os.path, bz2, pickle
    import dateutil
    import calendar

    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "field_ev_prefix" with unique instance ID
    # and filename to save new field data
    field_ev_prefix = "ev_field_lgbm_"
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
    # filter_filename = trainfile   # filter columns are in trainfile which must be specified in Constants - deprecated
    
    # fields matching the specified prefix will not be used in the model
    ignore_columns_containing = "{ignore_columns_containing}"
    # include only fields matching string e.g., only properly scaled columns should be used with MLP
    include_columns_containing = "{include_columns_containing}"
    
    objective_multiclass = {objective_multiclass}
    objective_regression = {objective_regression}
   
    def __init__(self):
        from datetime import datetime
        # remove the target field for this instance from the data used for training
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)
        
        # if saved models for the target field already exist then load them from filesystem
        self.predictors = []
        for fold in range(self.start_fold, self.nfolds):
            if self.os.path.isfile(workdir + self.output_column + "_fold" + str(fold) + ".model"):
                predictor_stored = self.lgb.Booster(model_file=workdir + self.output_column + "_fold" + str(fold) + ".model")
                self.predictors.append(predictor_stored)
                print (str(datetime.now()), self.output_column + ' fold ' + str(fold) + ' predictor model loaded')
  
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
    
    #def plot_feature_importance(self, n_top_features=20, graph_width=10, graph_height=25, importance_type='gain'):
        # this method can be used in Jupyter notebook to plot features of a particular model created by AIOS
        # copy whole DNA code as executed by AIOS into notebook with global Constants, initialise/run the class first
        #%matplotlib inline
        #self.lgb.plot_importance(self.bst, max_num_features=n_top_features, importance_type=importance_type).figure.set_size_inches(graph_width,graph_height)

    def my_log_loss(self, a, b):
        eps = 1e-9
        sum1 = 0.0
        for k in range(0, len(a)):
            bx = min(max(b[k],eps), 1-eps)
            sum1 += 1.0 * a[k] * self.math.log(bx) + 1.0 * (1 - a[k]) * self.math.log(1 - bx)
        return -sum1/len(a)

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
        for fold in range(self.start_fold, self.nfolds):
            pred += self.predictors[fold-self.start_fold].predict(df)
        # average prediction over all folds    
        pred = pred / (self.nfolds - self.start_fold)
        df_add[self.output_column] = pred

    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        global trainfile
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

        # rename columns in df to unique names
        df.columns = columns_new
        print (str(datetime.now()), " data loaded", len(df), "rows; ", len(df.columns), "columns")
        original_row_count = len(df)
        
        # analyse target column whether it is binary which may result in different loss function used
        target_classes = df[df[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()
        is_binary = target_classes==[0, 1]

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
            
        # prepare LGBM parameters    
        params = {}
        params['learning_rate'] = {learning_rate}       # shrinkage_rate
        params['boosting_type'] = {boosting_type}
        params['sub_feature'] = {sub_feature}           # feature_fraction (small values => use very different submodels)
        params['bagging_fraction'] = {bagging_fraction} # sub_row
        params['bagging_freq'] = {bagging_freq}
        params['num_leaves'] =     {num_leaves}            # num_leaf
        params['tree_learner'] = {tree_learner}
        params['min_data'] = {min_data}                 # min_data_in_leaf
        params['verbose'] = 1
        params['feature_fraction_seed'] = {feature_fraction_seed}
        params['bagging_seed'] = {bagging_seed}
        params['max_depth'] = {max_depth}
        params['num_threads'] = {num_threads}
        params['boost_from_average'] = {boost_from_average}
        params['is_unbalance'] = {is_unbalance}
        params['lambda_l1'] = {lambda_l1}
        params['lambda_l2'] = {lambda_l2}

        if is_binary:
            print ("detected binary target: use AUC/LOGLOSS")
            params['objective'] = 'binary'
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
            print ("\nFOLD", fold, "\n")
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

            y_train = x_train[self.target_col]                    # separate training fields and the target
            x_train = x_train.drop(self.target_col, 1)

            y_test = x_test[self.target_col]
            x_test = x_test.drop(self.target_col, 1)

            x_train = self.lgb.Dataset( x_train, label=y_train)    # convert DF to lgb.Dataset as required by LGBM

            num_round=100000
            watchlist  = [self.lgb.Dataset(x_test, label=y_test)]
            predictor = self.lgb.train( params, x_train, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )
            self.bst = predictor  # save trained model as class attribute, so e.g., plot_feature_importance can be called
            
            if mode==1:
                predictor.save_model(workdir + self.output_column + "_fold" + str(fold) + ".model")

            pred = predictor.predict(x_test)
            if is_binary:
                result = self.my_log_loss(y_test, pred)
                # show various metrics as per
                # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                result_roc_auc = roc_auc_score(y_test, pred)
                result_cm = confusion_matrix(y_test, (pred>0.5))  # assume 0.5 probability threshold
                result_cr = classification_report(y_test, (pred>0.5))
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
                result = predictor.best_score['valid_0']['multi_logloss']
                result_roc_auc = f1_score(y_test, pred_classes, average='weighted')
            else:
                result = sum(abs(y_test-pred))/len(y_test)
                #result = sqrt(mean_squared_error(y_test, pred))
                
            print ("result: ", result)
            
            weighted_result += result * len(pred)
            weighted_auc += result_roc_auc * len(pred)
            count_records_notnull += len(pred)

            # predict all examples in the original test set which may include erroneous examples previously removed
            #pred_all_test = predictor.predict(self.lgb.Dataset(x_test_orig.drop(self.target_col, axis=1)))
            pred_all_test = predictor.predict(x_test_orig.drop(self.target_col, axis=1))
            #prediction = self.np.concatenate([prediction,pred_all_test])
 
            if not is_binary and self.is_set(self.objective_multiclass):
                prediction[range_start:range_end] = self.np.argmax(pred_all_test, axis=1)
            else:
                prediction[range_start:range_end] = pred_all_test

            # predict validation and remainder sets examples
            if use_validation_set:
                predicted_valid_set += predictor.predict(self.df_valid.drop(self.target_col, axis=1))
                predicted_test_set += predictor.predict(df_test.drop(self.target_col, axis=1))
                
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
            if not is_binary and self.is_set(self.objective_multiclass):             
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
            elif self.is_set(self.objective_multiclass):
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
