import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

col_definition1 = "{random_field_numeric}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]

result_id = {id}
field_prefix = "field_"
output_column = field_prefix + str(result_id)
output_filename = output_column + ".csv"

df = pd.read_csv(workdir+file1)[[col1]]

scaler = StandardScaler()
df[output_column] = scaler.fit_transform(np.array(df[col1]).reshape(-1, 1))
df[[output_column]].to_csv(workdir+output_filename)

print ("StandardScaler("+col1+")")
print ("#add_field:"+output_column+",N,"+output_filename)
