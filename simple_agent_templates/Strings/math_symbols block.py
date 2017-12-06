import pandas as pd

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]

result_id = {id}

field_prefix = 'msym_'

df = pd.read_csv(workdir+file1)[[col1]]

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)

fldprefix = field_prefix + str(result_id)


df[fldprefix + '_1'] = df[col1].apply(lambda x: ('*' in set(x))+0 if type(type)==str else 0)
print (1)
df[fldprefix + '_2'] = df[col1].apply(lambda x: ('+' in set(x))+0 if type(type)==str else 0 )
print (2)
df[fldprefix + '_3'] = df[col1].apply(lambda x: ('-' in set(x))+0 if type(type)==str else 0 )
print (3)
df[fldprefix + '_4'] = df[col1].apply(lambda x: ('/' in set(x))+0 if type(type)==str else 0 )
print (4)
df[fldprefix + '_5'] = df[col1].apply(lambda x: ('%' in set(x))+0 if type(type)==str else 0 )
print (5)
df[fldprefix + '_6'] = df[col1].apply(lambda x: ('#' in set(x))+0 if type(type)==str else 0 )
print (6)
df[fldprefix + '_7'] = df[col1].apply(lambda x: ('@' in set(x))+0 if type(type)==str else 0 )
print (7)
df[fldprefix + '_8'] = df[col1].apply(lambda x: ('^' in set(x))+0 if type(type)==str else 0 )
print (8)
df[fldprefix + '_9'] = df[col1].apply(lambda x: ('!' in set(x))+0 if type(type)==str else 0 )
print (9)

total_cols = 9

nrow = len(df)

for i in range(1,total_cols+1):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    df[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname+","+str(nrow))
