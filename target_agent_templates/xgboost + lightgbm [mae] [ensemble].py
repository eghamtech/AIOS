#start_of_genes_definitions
#key=max_depth;  type=random_int;  from=2;  to=12;  step=2
#key=eta;  type=random_float;  from=0.003;  to=0.03;  step=0.0015
#key=fields_to_use;  type=random_int;  from=20;  to=300;  step=1
#key=data;  type=random_array_of_fields;  length=300
#key=colsample_bytree;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=subsample;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=u_limit;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=u_limit_apply;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=l_limit;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=l_limit_apply;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=u_limit_gbm;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=u_limit_apply_gbm;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=l_limit_gbm;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=l_limit_apply_gbm;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=max_depth2;  type=random_int;  from=2;  to=12;  step=2
#key=eta2;  type=random_float;  from=0.003;  to=0.03;  step=0.0015
#key=colsample_bytree2;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=subsample2;  type=random_float;  from=0.4;  to=1;  step=0.05
#key=u_limit2;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=u_limit2_apply;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=l_limit2;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=l_limit2_apply;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=learning_rate;  type=random_float;  from=0.001;  to=0.06;  step=0.001
#key=sub_feature;  type=random_float;  from=0.2;  to=1;  step=0.01
#key=bagging_fraction;  type=random_float;  from=0.2;  to=1;  step=0.01
#key=bagging_freq;  type=random_int;  from=10;  to=100;  step=1
#key=num_leaves;  type=random_int;  from=512;  to=4096;  step=5
#key=min_data;  type=random_int;  from=10;  to=2000;  step=5
#key=feature_fraction_seed;  type=random_int;  from=1;  to=10;  step=1
#key=bagging_seed;  type=random_int;  from=1;  to=10;  step=1
#key=boost_from_average;  type=random_from_set;  set=True,False
#key=k_xgb1;  type=random_float;  from=0;  to=1;  step=0.01
#key=k_xgb2;  type=random_float;  from=0;  to=1;  step=0.01
#key=k_gbm;  type=random_float;  from=0;  to=1;  step=0.01
#end_of_genes_definitions

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import xgboost as xgb
import lightgbm as lgb

data_defs = {data}

main_data = pd.read_csv(workdir+trainfile)

df = main_data[[target]]

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

x_test = df.copy(deep=True).iloc[1::2]
x_test.reset_index(drop=True, inplace=True)
x_train = df.copy(deep=True).iloc[0::2]
x_train.reset_index(drop=True, inplace=True)

print ("train " + target + " mean:", x_train[target].mean())
x_train.loc[x_train[target]>{u_limit}, target] = {u_limit_apply}
x_train.loc[x_train[target]<{l_limit}, target] = {l_limit_apply}
print ("train " + target + " mean:", x_train[target].mean())
            
print ("x_test rows count: " + str(len(x_test)))
print ("x_train rows count: " + str(len(x_train)))

y_train = x_train[target]
x_train = x_train.drop(target, 1)

y_test = x_test[target]
x_test = x_test.drop(target, 1)

dtrain = xgb.DMatrix( x_train, label=y_train)
dtest = xgb.DMatrix( x_test)

num_round=10000
watchlist  = [(dtrain,'train'), (xgb.DMatrix( x_test, label=y_test), 'test')]
param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'reg:linear', 'eval_metric':'mae', 'nthread':4}
predictor_xgb1 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )

prediction_xgb1 = predictor_xgb1.predict(dtest)

result = sum(abs(y_test-prediction_xgb1))/len(y_test)

print ("XGB-1 done with result:", result)



x_train = df.copy(deep=True).iloc[0::2]
x_train.reset_index(drop=True, inplace=True)

print ("train " + target + " mean:", x_train[target].mean())
x_train.loc[x_train[target]>{u_limit2}, target] = {u_limit2_apply}
x_train.loc[x_train[target]<{l_limit2}, target] = {l_limit2_apply}
print ("train " + target + " mean:", x_train[target].mean())
            
y_train = x_train[target]
x_train = x_train.drop(target, 1)

dtrain = xgb.DMatrix( x_train, label=y_train)

num_round=10000
watchlist  = [(dtrain,'train'), (xgb.DMatrix( x_test, label=y_test), 'test')]
param = {'max_depth':{max_depth2}, 'eta':{eta2}, 'colsample_bytree':{colsample_bytree2}, 'subsample': {subsample2}, 'objective':'reg:linear', 'eval_metric':'mae', 'nthread':4}
predictor_xgb2 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )

prediction_xgb2 = predictor_xgb2.predict(dtest)

result = sum(abs(y_test-prediction_xgb2))/len(y_test)

print ("XGB-2 done with result:", result)




params = {}
params['learning_rate'] = {learning_rate}
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'
params['sub_feature'] = {sub_feature}
params['bagging_fraction'] = {bagging_fraction}
params['bagging_freq'] = {bagging_freq}
params['num_leaves'] = 	{num_leaves}
params['min_data'] = {min_data}
params['verbose'] = 1
params['feature_fraction_seed'] = {feature_fraction_seed}
params['bagging_seed'] = {bagging_seed}
params['max_depth'] = -1
params['num_threads'] = 4
params['boost_from_average'] = {boost_from_average}

x_train = df.copy(deep=True).iloc[0::2]
x_train.reset_index(drop=True, inplace=True)

print ("train " + target + " mean:", x_train[target].mean())
x_train.loc[x_train[target]>{u_limit_gbm}, target] = {u_limit_apply_gbm}
x_train.loc[x_train[target]<{l_limit_gbm}, target] = {l_limit_apply_gbm}
print ("train " + target + " mean:", x_train[target].mean())
            
y_train = x_train[target]
x_train = x_train.drop(target, 1)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_test, label=y_test)
watchlist = [d_valid]
print("\nFitting LightGBM model ...")
predictor_gbm = lgb.train(params, d_train, 10000, watchlist, verbose_eval = 100, early_stopping_rounds=100)

prediction_gbm = predictor_gbm.predict(x_test)
result = sum(abs(y_test-prediction_gbm))/len(y_test)

print ("GBM done with result:", result)


ensemble = ({k_xgb1}*prediction_xgb1 + {k_xgb2}*prediction_xgb2 + {k_gbm}*prediction_gbm)/({k_xgb1} + {k_xgb2} + {k_gbm})

result = sum(abs(y_test-ensemble))/len(y_test)

print ("fitness="+str(result))

