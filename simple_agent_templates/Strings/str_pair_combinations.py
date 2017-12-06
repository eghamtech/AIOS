import pandas as pd

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}

field_prefix = 'comb_'

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)


##################################################

fldprefix = field_prefix + str(result_id)

block = int(len(df)/50)



qcomb = {}
k = block - 1
for index, row in df.iterrows():
    k+=1
    if type(row[col1])==str:
        q1 = row[col1]
    else:
        q1 = ''
    if type(row[col2])==str:
        q2 = row[col2]
    else:
        q2 = ''
    
    if q1 in qcomb:
        qcomb[q1][q2] = 1
    else:
        qcomb[q1] = {q2 : 1}

    if q2 in qcomb:
        qcomb[q2][q1] = 1
    else:
        qcomb[q2] = {q1 : 1}

    if k >= block:
        print (index)
        k=0


k = block-1
for index, row in df.iterrows():
    k+=1
    if type(row[col1])==str:
        sline1 = row[col1]
    else:
        sline1 = ''
    if type(row[col2])==str:
        sline2 = row[col2]
    else:
        sline2 = ''
    
    
    res = 0
    res += len(qcomb[sline1])
    qs1 = set(qcomb[sline1].keys())
        
    res += len(qcomb[sline2])
    qs2 = set(qcomb[sline2].keys())

    res2 = len(qs1.intersection(qs2))
    
    df.loc[index, fldprefix + '_1'] = res
    df.loc[index, fldprefix + '_2'] = res2
        
    if k>=block:
        k=0
        print (index)


total_cols = 2

nrow = len(df)
for i in range(1,total_cols+1):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    df[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

