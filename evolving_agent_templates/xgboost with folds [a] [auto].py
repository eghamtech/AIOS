#start_of_genes_definitions
#key=colsample_bytree;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=subsample;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=fields_to_use;  type=random_int;  from=40;  to=400;  step=1
#key=data;  type=random_array_of_fields;  length=400
#key=eta;  type=random_float;  from=0.03;  to=0.3;  step=0.01
#key=max_depth;  type=random_int;  from=6;  to=14;  step=2
#key=nfolds;  type=random_int;  from=10;  to=10;  step=1
#key=use_validation_set;  type=random_from_set;  set=True
#key=filter_column;  type=random_from_set;  set=id
#key=validation_set_start_value;  type=random_from_set;  set=350000
#end_of_genes_definitions

class cls_ev_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import xgboost as xgb
    import numpy as np
    import math
    import os.path
    import dateutil
    import calendar

    result_id = {id}
    field_ev_prefix = "ev_field_"
    output_column = field_ev_prefix + str(result_id)
    output_filename = output_column + ".csv"

    target_definition = "{field_to_predict}"
    target_col = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]

    data_defs = {data}
    
    filter_column = "{filter_column}"
    filter_filename = trainfile

    def __init__(self):
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)
        
        if self.os.path.isfile(workdir + self.output_column + ".model"):
            self.predictor_stored = self.xgb.Booster()
            self.predictor_stored.load_model(workdir + self.output_column + ".model")

    def timestamp(self, x):
        return self.calendar.timegm(self.dateutil.parser.parse(x).timetuple())
    
    def plot_feature_importance(self, n_top_features=20, graph_width=10, graph_height=25):
        # this method can be used in Jupyter notebook to plot features of a particular model created by AIOS
        # copy whole DNA code as executed by AIOS into notebook with global Constants, initialise/run the class first
        %matplotlib inline
        self.xgb.plot_importance(self.bst, max_num_features=n_top_features).figure.set_size_inches(graph_width,graph_height)

    def my_log_loss(self, a, b):
        eps = 1e-9
        sum1 = 0.0
        for k in range(0, len(a)):
            bx = min(max(b[k],eps), 1-eps)
            sum1 += 1.0 * a[k] * self.math.log(bx) + 1.0 * (1 - a[k]) * self.math.log(1 - bx)
        return -sum1/len(a)

    def apply(self, df_add):
        columns_new = []
        columns = []
        
        cols_count = 0
        for i in range(0,len(self.data_defs)):
            cols_count+=1
            if cols_count>{fields_to_use}:
                break
            col_name = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if i==0:
                df = df_add[[col_name]]
            else:
                df = df.merge(df_add[[col_name]], left_index=True, right_index=True)
            
            columns.append(col_name)
            ncol_count = columns.count(col_name)
            if ncol_count==1:
                columns_new.append(col_name)
            else:
                columns_new.append(col_name+"_v"+str(ncol_count))
        
        df.columns = columns_new
        dtest = self.xgb.DMatrix(df)
        pred = self.predictor_stored.predict(dtest)
        df_add[self.output_column] = pred

    def run(self, mode):
        global trainfile
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        print ("enter run mode " + str(mode))  # 0=work for fitness only;  1=make new output field

        use_validation_set = {use_validation_set}
        
        if use_validation_set:
            df_filter_column = self.pd.read_csv(workdir+self.filter_filename, usecols = [self.filter_column])
            use_indexes = df_filter_column[df_filter_column[self.filter_column]<{validation_set_start_value}].index
            print ("Length of train set:", len(use_indexes), ", length of validation set:", len(df_filter_column)-len(use_indexes))
        
        df = self.pd.read_csv(workdir+self.target_file)[[self.target_col]] #main_data[[target]]

        columns_new = [self.target_col]
        columns = [self.target_col]

        cols_count = 0
        for i in range(0,len(self.data_defs)):
            cols_count+=1
            if cols_count>{fields_to_use}:
                break
            col_name = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            df = df.merge(self.pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)

            columns.append(col_name)
            ncol_count = columns.count(col_name)
            if ncol_count==1:
                columns_new.append(col_name)
            else:
                columns_new.append(col_name+"_v"+str(ncol_count))

        df.columns = columns_new
        print ("data loaded", len(df), "rows; ", len(df.columns), "columns")
        is_binary = df[df[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()==[0, 1]

        if use_validation_set:
            df_valid = df[df.index.isin(use_indexes)==False]
            df_valid.reset_index(drop=True, inplace=True)
            df = df[df.index.isin(use_indexes)]
            df.reset_index(drop=True, inplace=True)
            predicted_valid_set = self.np.zeros(len(df_valid))
            
        if is_binary:
            print ("detected binary target. use LOGLOSS")
            param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'binary:logistic', 'eval_metric':'logloss', 'nthread':4}
        else:
            print ("use MAE")
            param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'reg:linear', 'eval_metric':'mae', 'nthread':4}

        #############################################################
        #
        #                   MAIN LOOP
        #
        #############################################################

        nfolds = {nfolds}
        block = int(len(df)/nfolds)

        prediction = []

        weighted_result = 0
        count_records_notnull = 0

        for fold in range(0,nfolds):
            print ("\nFOLD", fold, "\n")
            range_start = fold*block
            range_end = (fold+1)*block
            if fold==nfolds-1:
                range_end = len(df)
            range_predict = range(range_start, range_end)
            print ("range start", range_start, "; range end ", range_end)

            x_test = df[df.index.isin(range_predict)]
            x_test.reset_index(drop=True, inplace=True)
            x_test_orig = x_test.copy()
            x_test = x_test[x_test[self.target_col].notnull()]
            x_test.reset_index(drop=True, inplace=True)

            x_train = df[df.index.isin(range_predict)==False]
            x_train.reset_index(drop=True, inplace=True)
            x_train= x_train[x_train[self.target_col].notnull()]
            x_train.reset_index(drop=True, inplace=True)

            print ("x_test rows count: " + str(len(x_test)))
            print ("x_train rows count: " + str(len(x_train)))

            y_train = x_train[self.target_col]
            x_train = x_train.drop(self.target_col, 1)

            y_test = x_test[self.target_col]
            x_test = x_test.drop(self.target_col, 1)

            dtrain = self.xgb.DMatrix( x_train, label=y_train)
            dtest = self.xgb.DMatrix( x_test)

            num_round=100000
            watchlist  = [(dtrain,'train'), (self.xgb.DMatrix( x_test, label=y_test), 'test')]
            predictor = self.xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=10 )
            self.bst = predictor  # save trained model as class attribute, so e.g., plot_feature_importance can be called
            
            if mode==1 and fold==nfolds-1:
                predictor.save_model(workdir + self.output_column + ".model")

            pred = predictor.predict(dtest)
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
            else:
                result = sum(abs(y_test-pred))/len(y_test)

            print ("result: ", result)
            
            weighted_result += result * len(pred)
            count_records_notnull += len(pred)

            # this must be tested once more, may be this actions must be done only for the last fold???
            pred_all_test = predictor.predict(self.xgb.DMatrix(x_test_orig.drop(self.target_col, axis=1)))
            prediction = self.np.concatenate([prediction,pred_all_test])

            if use_validation_set:
                predicted_valid_set += predictor.predict(self.xgb.DMatrix(df_valid.drop(self.target_col, axis=1)))
                
        weighted_result = weighted_result/count_records_notnull
        print ("weighted_result:", weighted_result)

        if use_validation_set:
            print()
            print()
            print ("*************  VALIDATION SET RESULTS  *****************")
            if is_binary:
                predicted_valid_set = predicted_valid_set / nfolds
                y_valid = df_valid[self.target_col]
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
                result = sum(abs(y_valid-predicted_valid_set))/len(y_valid)
                print ("MAE: ", result)
                
        #############################################################
        #
        #                   OUTPUT
        #
        #############################################################

        if mode==1:
            df[self.output_column] = prediction
            df[[self.output_column]].to_csv(workdir+self.output_filename)

            nrow = len(df)
            print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(nrow))
        else:
            print ("fitness="+str(weighted_result))

ev_agent_{id} = cls_ev_agent_{id}()
