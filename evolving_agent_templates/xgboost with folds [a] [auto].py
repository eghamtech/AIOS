#start_of_genes_definitions
#key=colsample_bytree;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=subsample;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=fields_to_use;  type=random_int;  from=40;  to=70;  step=1
#key=data;  type=random_array_of_fields;  length=70
#key=eta;  type=random_float;  from=0.1;  to=0.3;  step=0.01
#key=max_depth;  type=random_int;  from=6;  to=14;  step=2
#key=nfolds;  type=random_int;  from=2;  to=2;  step=1
#end_of_genes_definitions

class cls_ev_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import xgboost as xgb
    import numpy as np
    import math
    import os.path

    result_id = {id}
    field_ev_prefix = "ev_field_"
    output_column = field_ev_prefix + str(result_id)
    output_filename = output_column + ".csv"

    target_definition = "{field_to_predict}"
    target_col = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]

    data_defs = {data}

    def __init__(self):
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)
        
        if self.os.path.isfile(workdir + self.output_column + ".model"):
            self.predictor_stored = self.xgb.Booster()
            self.predictor_stored.load_model(workdir + self.output_column + ".model")

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
            
        dtest = self.xgb.DMatrix(df)
        pred = self.predictor_stored.predict(dtest)
        df_add[self.output_column] = pred

    def run(self, mode):
        global trainfile
        from sklearn.metrics import roc_auc_score
        print ("enter run mode " + str(mode))  # 0=work for fitness only;  1=make new output field

        main_data = self.pd.read_csv(workdir+trainfile)
        df = self.pd.read_csv(workdir+self.target_file)[[self.target_col]] #main_data[[target]]

        cols_count = 0
        for i in range(0,len(self.data_defs)):
            cols_count+=1
            if cols_count>{fields_to_use}:
                break
            col_name = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if file_name==trainfile:
                df[col_name] = main_data[col_name]
            else:
                df = df.merge(self.pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)

        print ("data loaded", len(df), "rows; ", len(df.columns), "columns")
        is_binary = df[df[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()==[0, 1]

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
                result_roc_auc = roc_auc_score(y_test, pred)
            else:
                result = sum(abs(y_test-pred))/len(y_test)

            print ("result: ", result)
            print ("ROC AUC score: ", result_roc_auc)
            weighted_result += result * len(pred)
            count_records_notnull += len(pred)

            pred_all_test = predictor.predict(self.xgb.DMatrix(x_test_orig.drop(self.target_col, axis=1)))

            prediction = self.np.concatenate([prediction,pred_all_test])

        weighted_result = weighted_result/count_records_notnull
        print ("weighted_result:", weighted_result)

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
