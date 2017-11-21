#start_of_genes_definitions
#key=max_depth;			type=random_int;	from=2;		to=12;		step=2
#key=eta;				type=random_float;	from=0.003;	to=0.03;	step=0.0015
#key=fields_to_use;		type=random_int;	from=20;	to=300;		step=1
#key=data;				type=random_array_of_fields;	length=300
#key=colsample_bytree;	type=random_float;	from=0.4;	to=1;		step=0.05
#key=subsample;			type=random_float;	from=0.4;	to=1;		step=0.05
#key=u_limit;			type=random_float;	from=0.01;	to=0.4;		step=0.001
#key=u_limit_apply;		type=random_float;	from=0.01;	to=0.4;		step=0.001
#key=l_limit;			type=random_float;	from=-0.4;	to=-0.01;	step=0.001
#key=l_limit_apply;		type=random_float;	from=-0.4;	to=-0.01;	step=0.001
#end_of_genes_definitions

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import xgboost as xgb

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

param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'reg:linear', 'eval_metric':'mae', 'nthread':4}

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
predictor = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )

prediction = predictor.predict(dtest)
result = sum(abs(y_test-prediction))/len(y_test)

print ("fitness="+str(result))

