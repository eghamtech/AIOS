#start_of_parameters
#key=DNA_ID;  type=constant;  value=12345
#key=outfile;  type=constant;  value=submission.csv
#key=default_num_round;  type=constant;  value=1000
#end_of_parameters


import pandas as pd

default_num_round = {default_num_round}
outfile = "{outfile}"

{run_dna {DNA_ID}}

df = pd.read_csv(workdir+trainfile)
df = df[df[target].isnull()]
df.reset_index(drop=True, inplace=True)
df[target] = prediction

df.to_csv(workdir+outfile, index=False)

print ("Output agent finished")
