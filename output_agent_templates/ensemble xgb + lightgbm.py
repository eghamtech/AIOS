#start_of_parameters
#key=outfile;  type=constant;  value=submission.csv
#key=use_input_file;  type=constant;  value=train_quora.csv
#key=DNA_ID_xgb;  type=constant;  value=123
#key=DNA_ID_lgbm;  type=constant;  value=123
#key=num_round_xgb;  type=constant;  value=1234
#key=num_round_lgbm;  type=constant;  value=1234
#end_of_parameters


import pandas as pd

pd.set_option('display.width', 500)

outfile = "{outfile}"
input_file = "{use_input_file}"

df_out = pd.read_csv(workdir+input_file)
df_out = df_out[df_out[target].isnull()]
df_out.reset_index(drop=True, inplace=True)

default_num_round = {num_round_xgb}
{run_dna {DNA_ID_xgb}}
df_out["p_xgb"] = prediction


default_num_round = {num_round_lgbm}
{run_dna {DNA_ID_lgbm}}
df_out["p_lgbm"] = prediction

k1 = 0.5
k2 = 0.5
df_out[target] = (k1 * df_out["p_xgb"] + k2 * df_out["p_lgbm"]) / (k1 + k2)


df_out.to_csv(workdir+outfile, index=False)

print ("Output agent finished")
print (df_out)
