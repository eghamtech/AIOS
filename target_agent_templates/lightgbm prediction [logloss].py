#start_of_genes_definitions
#key=fields_to_use;  type=random_int;  from=20;  to=100;  step=1
#key=data;  type=random_array_of_fields;  length=100
#key=u_limit;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=u_limit_apply;  type=random_float;  from=0.01;  to=0.4;  step=0.001
#key=l_limit;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=l_limit_apply;  type=random_float;  from=-0.4;  to=-0.01;  step=0.001
#key=learning_rate;  type=random_float;  from=0.001;  to=0.06;  step=0.001
#key=sub_feature;  type=random_float;  from=0.2;  to=1;  step=0.01
#key=bagging_fraction;  type=random_float;  from=0.2;  to=1;  step=0.01
#key=bagging_freq;  type=random_int;  from=10;  to=100;  step=1
#key=num_leaves;  type=random_int;  from=512;  to=4096;  step=5
#key=min_data;  type=random_int;  from=100;  to=2000;  step=5
#key=feature_fraction_seed;  type=random_int;  from=1;  to=10;  step=1
#key=bagging_seed;  type=random_int;  from=1;  to=10;  step=1
#key=boost_from_average;  type=random_from_set;  set=True,False
#end_of_genes_definitions

if not ('output_mode' in locals()):
    output_mode = False

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import lightgbm as lgb
import math

def my_log_loss(a, b):
    eps = 1e-9
    sum1 = 0.0
    for k in range(0, len(a)):
        bx = min(max(b[k],eps), 1-eps)
        sum1 += 1.0 * a[k] * math.log(bx) + 1.0 * (1 - a[k]) * math.log(1 - bx)
    return -sum1/len(a)

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

params = {}
params['learning_rate'] = {learning_rate} # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = {sub_feature}    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = {bagging_fraction} # sub_row
params['bagging_freq'] = {bagging_freq}
params['num_leaves'] = 	{num_leaves}        # num_leaf
params['min_data'] = {min_data}         # min_data_in_leaf
params['verbose'] = 1
params['feature_fraction_seed'] = {feature_fraction_seed}
params['bagging_seed'] = {bagging_seed}
params['max_depth'] = -1
params['num_threads'] = 4
params['boost_from_average'] = {boost_from_average}

if not output_mode:
    x_test = df.copy(deep=True).iloc[1::2]
    x_test = x_test[x_test[target].notnull()]
    x_test.reset_index(drop=True, inplace=True)
    x_train = df.copy(deep=True).iloc[0::2]
    x_train = x_train[x_train[target].notnull()]
    x_train.reset_index(drop=True, inplace=True)
    
    y_test = x_test[target]
    
    num_round = 10000
else:
    x_test = df[df[target].isnull()]
    x_test.reset_index(drop=True, inplace=True)
    x_train = df[df[target].notnull()]
    x_train.reset_index(drop=True, inplace=True)
    
    num_round = default_num_round
    
print ("train " + target + " mean:", x_train[target].mean())
x_train.loc[x_train[target]>{u_limit}, target] = {u_limit_apply}
x_train.loc[x_train[target]<{l_limit}, target] = {l_limit_apply}
print ("train " + target + " mean:", x_train[target].mean())
            
print ("x_test rows count: " + str(len(x_test)))
print ("x_train rows count: " + str(len(x_train)))

y_train = x_train[target]
x_train = x_train.drop(target, 1)

x_test = x_test.drop(target, 1)

d_train = lgb.Dataset(x_train, label=y_train)
if not output_mode:
    d_valid = lgb.Dataset(x_test, label=y_test)
else:
    d_valid = lgb.Dataset(x_train, label=y_train)

watchlist = [d_valid]
print("\nFitting LightGBM model ...")
predictor = lgb.train(params, d_train, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100)

prediction = predictor.predict(x_test)

if not output_mode:
    result = my_log_loss(y_test, prediction)
    print ("fitness="+str(result))

