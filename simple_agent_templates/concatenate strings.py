import pandas as pd

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}
output_column = field_prefix + str(result_id)
output_filename = output_column + ".csv"

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[output_column] = df[col1].map(dict1) + "," + df[col2].map(dict2)  #concatenate

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
    df[output_column] = df[output_column].map(dict_out)    #convert new column to dict
    pd.DataFrame(list(dict_out.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+output_column+'.csv')    #save new column dict
    df[[output_column]].to_csv(workdir+output_filename)
    print (col1+"+"+col2)
    print ("#add_field:"+output_column+",Y,"+output_filename)
