import pandas as pd

col_definition1 = "{random_field_numeric}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_field_numeric}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}
output_column = field_prefix + str(result_id)
output_filename = output_column + ".csv"

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

df[output_column] = df[col1] + df[col2]
df[[output_column]].to_csv(workdir+output_filename)

print (col1+"+"+col2)
print ("#add_field:"+output_column+",N,"+output_filename)
