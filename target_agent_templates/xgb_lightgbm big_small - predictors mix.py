#start_of_genes_definitions
#key=max_depth;				type=random_int;	from=10;	to=12;		step=1
#key=eta;					type=random_float;	from=0.006;	to=0.01;	step=0.0015
#key=colsample_bytree;		type=random_float;	from=0.4;	to=1;		step=0.05
#key=subsample;				type=random_float;	from=0.4;	to=1;		step=0.05
#key=u_limit;				type=random_float;	from=0.03;	to=0.1;		step=0.001
#key=l_limit;				type=random_float;	from=-0.1;	to=-0.03;	step=0.001
#key=u_limit2;				type=random_float;	from=0.03;	to=0.1;		step=0.001
#key=l_limit2;				type=random_float;	from=-0.1;	to=-0.03;	step=0.001
#key=learning_rate;			type=random_float;	from=0.006;	to=0.01;	step=0.0015
#key=sub_feature;			type=random_float;	from=0.2;	to=1;		step=0.01
#key=bagging_fraction;		type=random_float;	from=0.2;	to=1;		step=0.01
#key=bagging_freq;			type=random_int;	from=10;	to=100;		step=1
#key=num_leaves;			type=random_int;	from=512;	to=4096;	step=16
#key=min_data;				type=random_int;	from=100;	to=2000;	step=5
#key=feature_fraction_seed;	type=random_int;	from=1;		to=10;		step=1
#key=bagging_seed;			type=random_int;	from=1;		to=10;		step=1
#key=boost_from_average;	type=random_from_set;			set=True,False
#key=lowrange;				type=random_float;	from=0.01;	to=0.08;	step=0.005
#key=repeat_lowrange;		type=random_int;	from=1;		to=5;		step=1
#key=pbig1;					type=random_float;	from=0.01;	to=0.1;		step=0.001
#key=pbig2;					type=random_float;	from=0.04;	to=0.12;	step=0.001
#key=pbig3;					type=random_float;	from=0.08;	to=0.3;		step=0.001
#key=pbig4;					type=random_float;	from=0.16;	to=0.5;		step=0.001
#key=k1;					type=random_float;	from=0;		to=1;		step=0.01
#key=k2;					type=random_float;	from=0;		to=1;		step=0.01
#key=k3;					type=random_float;	from=0;		to=1;		step=0.01
#key=k4;					type=random_float;	from=0;		to=1;		step=0.01
#key=ktotal;				type=random_float;	from=0.5;	to=1.5;		step=0.01
#key=border1;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=border1a;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=border2;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=border2a;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=border2b;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=border3;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=border3a;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=border4;				type=random_float;	from=0.2;	to=1;		step=0.01
#key=value1p;				type=random_float;	from=0.01;	to=1;		step=0.001
#key=value2p;				type=random_float;	from=0.01;	to=1;		step=0.001
#key=value3p;				type=random_float;	from=0.01;	to=1;		step=0.001
#key=value4p;				type=random_float;	from=0.01;	to=1;		step=0.001
#key=value1n;				type=random_float;	from=-1;	to=-0.01;	step=0.001
#key=value2n;				type=random_float;	from=-1;	to=-0.01;	step=0.001
#key=value3n;				type=random_float;	from=-1;	to=-0.01;	step=0.001
#key=value4n;				type=random_float;	from=-1;	to=-0.01;	step=0.001
#end_of_genes_definitions

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np

df = pd.read_csv(workdir+trainfile)

df['N-ValueRatio'] = (df['taxvaluedollarcnt']/df['taxamount'])
df['N-LivingAreaProp'] = (df['calculatedfinishedsquarefeet']/df['lotsizesquarefeet'])
df['N-ValueProp'] = (df['structuretaxvaluedollarcnt']/df['landtaxvaluedollarcnt'])
df["N-location"] = df["latitude"] + df["longitude"]
df["N-location-2"] = df["latitude"]*df["longitude"]

print ("data loaded", len(df), "rows; ", len(df.columns), "columns")

param = {'max_depth':{max_depth}, 'eta':{eta}, 'colsample_bytree':{colsample_bytree}, 'subsample': {subsample}, 'objective':'reg:linear', 'eval_metric':'mae', 'nthread':4}
params = {}
params['learning_rate'] = {learning_rate}
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
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

for method in [1,2,3,4,5,6,7,8]:
    print ("\n\nMETHOD", method, "\n\n")
    x_test = df.copy(deep=True).iloc[1::2]
    x_test.reset_index(drop=True, inplace=True)
    x_train = df.copy(deep=True).iloc[0::2]
    x_train.reset_index(drop=True, inplace=True)

    if method in [1,2]:
        x_train.loc[x_train[target]>{u_limit}, target] = {u_limit}
        x_train.loc[x_train[target]<{l_limit}, target] = {l_limit}
    elif method in [3,4]:
        x_train.loc[x_train[target]>{u_limit2}, target] = {u_limit2}
        x_train.loc[x_train[target]<{l_limit2}, target] = {l_limit2}
        df1 = x_train[x_train["logerror"].abs()<{lowrange}]
        for m in range(0,{repeat_lowrange}):
            x_train = x_train.append(df1)
    elif method==5:
        x_train[target] = x_train[target].apply(lambda x: 1 if x>{pbig1} else -1 if x<-{pbig1} else 0)
        x_test[target] = x_test[target].apply(lambda x: 1 if x>{pbig1} else -1 if x<-{pbig1} else 0)
    elif method==6:
        x_train[target] = x_train[target].apply(lambda x: 1 if x>{pbig2} else -1 if x<-{pbig2} else 0)
        x_test[target] = x_test[target].apply(lambda x: 1 if x>{pbig2} else -1 if x<-{pbig2} else 0)
    elif method==7:
        x_train[target] = x_train[target].apply(lambda x: 1 if x>{pbig3} else -1 if x<-{pbig3} else 0)
        x_test[target] = x_test[target].apply(lambda x: 1 if x>{pbig3} else -1 if x<-{pbig3} else 0)
    elif method==8:
        x_train[target] = x_train[target].apply(lambda x: 1 if x>{pbig4} else -1 if x<-{pbig4} else 0)
        x_test[target] = x_test[target].apply(lambda x: 1 if x>{pbig4} else -1 if x<-{pbig4} else 0)

    y_train = x_train[target]
    x_train = x_train.drop(target, 1)
    y_test = x_test[target]
    x_test = x_test.drop(target, 1)

    num_round=20000
    if method in [1,3,5,6,7,8]:
        dtrain = xgb.DMatrix( x_train, label=y_train)
        dtest = xgb.DMatrix( x_test)

        watchlist  = [(dtrain,'train'), (xgb.DMatrix( x_test, label=y_test), 'test')]
        
        if method==1:
            predictor1 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )
            prediction1 = predictor1.predict(dtest)
            y_test_main = y_test
        elif method==3:
            predictor3 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )
            prediction3 = predictor3.predict(dtest)
        elif method==5:
            predictor5 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )
            prediction5 = predictor5.predict(dtest)
        elif method==6:
            predictor6 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )
            prediction6 = predictor6.predict(dtest)
        elif method==7:
            predictor7 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )
            prediction7 = predictor7.predict(dtest)
        elif method==8:
            predictor8 = xgb.train( param, dtrain, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100 )
            prediction8 = predictor8.predict(dtest)
    else:
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_test, label=y_test)
        watchlist = [d_valid]

        if method==2:
            predictor2 = lgb.train(params, d_train, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100)
            prediction2 = predictor2.predict(x_test)
        elif method==4:
            predictor4 = lgb.train(params, d_train, num_round, watchlist, verbose_eval = 100, early_stopping_rounds=100)
            prediction4 = predictor4.predict(x_test)


print ("making final ensemble...")
sub = pd.DataFrame(y_test_main)
sub["ensemble"] = {ktotal} * ({k1}*prediction1 + {k2}*prediction2 + {k3}*prediction3 + {k4}*prediction4)/({k1} + {k2} + {k3} + {k4})
sub["prediction"] = sub["ensemble"]

sub.loc[np.logical_and(prediction5>{border1}, sub["ensemble"]>0), "prediction"] = {value1p}
sub.loc[np.logical_and(prediction6>{border2}, prediction5>{border1a}), "prediction"] = {value2p}
sub.loc[np.logical_and(prediction7>{border3}, prediction6>{border2a}), "prediction"] = {value3p}
sub.loc[np.logical_and(prediction8>{border4}, np.logical_and(prediction7>{border3a}, prediction6>{border2b})), "prediction"] = {value4p}

sub.loc[np.logical_and(prediction5<-{border1}, sub["ensemble"]<0), "prediction"] = {value1n}
sub.loc[np.logical_and(prediction6<-{border2}, prediction5<-{border1a}), "prediction"] = {value2n}
sub.loc[np.logical_and(prediction7<-{border3}, prediction6<-{border2a}), "prediction"] = {value3n}
sub.loc[np.logical_and(prediction8<-{border4}, np.logical_and(prediction7<-{border3a}, prediction6<-{border2b})), "prediction"] = {value4n}

result = 1.0*sum(abs(sub["prediction"]-sub[target]))/len(sub)

print ("fitness="+str(result))

