#start_of_genes_definitions
#key=colsample_bytree;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=subsample;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=fields_to_use;  type=random_int;  from=40;  to=70;  step=1
#key=data;  type=random_array_of_fields;  length=70
#key=eta;  type=random_float;  from=0.1;  to=0.3;  step=0.01
#key=max_depth;  type=random_int;  from=6;  to=14;  step=2
#key=nfolds;  type=random_int;  from=2;  to=2;  step=1
#end_of_genes_definitions

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import xgboost as xgb
import numpy as np
import math

result_id = {id}
field_ev_prefix = "ev_field_"
output_column = field_ev_prefix + str(result_id)
output_filename = output_column + ".csv"

target_definition = "{field_to_predict}"
target_col = target_definition.split("|")[0]
target_file = target_definition.split("|")[1]

data_defs = {data}

if target_definition in data_defs:
    data_defs.remove(target_definition)

#############################################################
#
#                   DATA PREPARATION
#
#############################################################

main_data = pd.read_csv(workdir+trainfile)

df = pd.read_csv(workdir+target_file)[[target_col]] #main_data[[target]]

cols_count = 0
for i in range(0,len(data_defs)):
    cols_count+=1
    if cols_count>{fields_to_use}:
        break
    col_name = data_defs[i].split("|")[0]
    file_name = data_defs[i].split("|")[1]
    
    if file_name==trainfile:
        df[col_name] = main_data[col_name]
    else:
        df = df.merge(pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)


print ("data loaded", len(df), "rows; ", len(df.columns), "columns")

import os.path
import sys
if output_mode==1:
    if os.path.isfile(workdir + output_filename):
        df_old = pd.read_csv(workdir + output_filename)
        if len(df)-len(df_old)==1: # incremental mode
            if os.path.isfile(workdir + output_column + ".model"):
                df[output_column] = df_old[output_column]
                predictor = xgb.Booster()
                predictor.load_model(workdir + output_column + ".model")
                x_test = df[-1:]
                x_test = x_test.drop(target_col, 1)
                x_test = x_test.drop(output_column, 1)
                dtest = xgb.DMatrix( x_test)
                pred = predictor.predict(dtest)
                nrow = len(df)
                df.at[nrow-1, output_column] = pred[0]
                df[[output_column]].to_csv(workdir+output_filename)
                print ("#add_field:"+output_column+",N,"+output_filename+","+str(nrow))
                sys.exit()

is_binary = df[df[target_col].notnull()].sort_values(target_col)[target_col].unique().tolist()==[0, 1]

if is_binary:
    print ("detected binary target. use LOGLOSS")
    param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'binary:logistic', 'eval_metric':'logloss', 'nthread':4}
else:
    print ("use MAE")
    param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'reg:linear', 'eval_metric':'mae', 'nthread':4}


def my_log_loss(a, b):
    eps = 1e-9
    sum1 = 0.0
    for k in range(0, len(a)):
        bx = min(max(b[k],eps), 1-eps)
        sum1 += 1.0 * a[k] * math.log(bx) + 1.0 * (1 - a[k]) * math.log(1 - bx)
    return -sum1/len(a)


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
    x_test = x_test[x_test[target_col].notnull()]
    x_test.reset_index(drop=True, inplace=True)

    x_train = df[df.index.isin(range_predict)==False]
    x_train.reset_index(drop=True, inplace=True)
    x_train= x_train[x_train[target_col].notnull()]
    x_train.reset_index(drop=True, inplace=True)

    print ("x_test rows count: " + str(len(x_test)))
    print ("x_train rows count: " + str(len(x_train)))

    y_train = x_train[target_col]
    x_train = x_train.drop(target_col, 1)

    y_test = x_test[target_col]
    x_test = x_test.drop(target_col, 1)

    dtrain = xgb.DMatrix( x_train, label=y_train)
    dtest = xgb.DMatrix( x_test)
    
    num_round=100000
    watchlist  = [(dtrain,'train'), (xgb.DMatrix( x_test, label=y_test), 'test')]
    predictor = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=10 )

    if output_mode==1 and fold==nfolds-1:
        predictor.save_model(workdir + output_column + ".model")
    
    pred = predictor.predict(dtest)
    if is_binary:
        result = my_log_loss(y_test, pred)
    else:
        result = sum(abs(y_test-pred))/len(y_test)

    print ("result:", result)
    weighted_result += result * len(pred)
    count_records_notnull += len(pred)
    
    pred_all_test = predictor.predict(xgb.DMatrix(x_test_orig.drop(target_col, axis=1)))
    
    prediction = np.concatenate([prediction,pred_all_test])

weighted_result = weighted_result/count_records_notnull
print ("weighted_result:", weighted_result)

#############################################################
#
#                   OUTPUT
#
#############################################################

if output_mode==1:
    df[output_column] = prediction
    df[[output_column]].to_csv(workdir+output_filename)

    nrow = len(df)
    print ("#add_field:"+output_column+",N,"+output_filename+","+str(nrow))
else:
    print ("fitness="+str(weighted_result))

