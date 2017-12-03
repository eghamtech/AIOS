# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# obtain random field of numerical type
col_definition1 = "{random_field_numeric}"
# field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
# load these two parts into variables
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]

# obtain a unique ID for the current instance
result_id = {id}
# create new field name with unique instance ID
# and filename to save new field data
field_prefix = "field_"
output_column = field_prefix + str(result_id)
output_filename = output_column + ".csv"

# "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
df = pd.read_csv(workdir+file1)[[col1]]

# use sklearn library to scale col1 in df, it needs to be reshaped and converted to Numpy Array first
scaler = StandardScaler()
df[output_column] = scaler.fit_transform(np.array(df[col1]).reshape(-1, 1))
df[[output_column]].to_csv(workdir+output_filename)

print ("StandardScaler("+col1+")")
print ("#add_field:"+output_column+",N,"+output_filename)
