# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

import pandas as pd

# obtain random field of string type
col_definition1 = "{random_dict}"
# field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
# load these two parts into variables
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]

# obtain another random field of string type and load field name and filename into variables
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

# obtain a unique ID for the current instance
result_id = {id}

# create new field name based on "field_prefix" (also specified in Constants) with unique instance ID
# and filename to save new field data
field_prefix = "field_"
output_column = field_prefix + str(result_id)
output_filename = output_column + ".csv"

# "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
# read both fields from respective CSV files
df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

# since both of the fields used in this agent are string based, CSV files loaded above contain only numerical indicies
# now read the secondary CSV files that hold the actual string data as we need it for concatenating such fields
dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

# concatenate source fields by mapping dataframe from the main CSV file to the secondary CSV file for each of the two fields and then joining 
df[output_column] = df[col1].map(dict1) + "," + df[col2].map(dict2) 

# function to create a column of numerical indicies based on column of string values
def make_dict(col):
    a1 = col.unique()
    a1 = [x for x in a1 if str(x) != 'nan']
    keys1 = range(1, len(a1)+1)
    return dict(zip(a1, keys1))

dict_out = make_dict(df[output_column])

if len(dict_out)==0:
    print("result column is empty")
    print ("#error")
else:
    # convert new column to dict (numerical indicies)
    df[output_column] = df[output_column].map(dict_out)
    # save new column dict
    pd.DataFrame(list(dict_out.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+output_column+'.csv')  
    df[[output_column]].to_csv(workdir+output_filename)
    # output field names into log
    print (col1+"+"+col2)
    # register new field/column in AIOS
    print ("#add_field:"+output_column+",Y,"+output_filename+","+str(len(df)))
