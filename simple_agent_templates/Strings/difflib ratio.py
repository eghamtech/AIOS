import pandas as pd
import difflib

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}

field_prefix = 'difflib_'

output_filename = field_prefix + str(result_id) + ".csv"

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(st1, st2)
    return seq.ratio()

func = lambda s: s[:1].lower() + s[1:] if s else ''



fldprefix = field_prefix + str(result_id)

block = int(len(df)/50)
i = block-1

for index, row in df.iterrows():
    i+=1
    if type(row[col1])==str:
        sline1 = func(row[col1])
    else:
        sline1 = ''
    if type(row[col2])==str:
        sline2 = func(row[col2])
    else:
        sline2 = ''
    
    df.set_value(index, fldprefix + '_1', diff_ratios(sline1, sline2))
    
    if i>=block:
        i=0
        print (index)

        
        
newfields = []

for i in range(1,2):
    newfields.append(fldprefix + '_' + str(i))
    
df[newfields].to_csv(workdir+output_filename)

nrow = len(df)

for fld in newfields:
    print ("#add_field:"+fld+",N,"+output_filename+","+str(nrow))

