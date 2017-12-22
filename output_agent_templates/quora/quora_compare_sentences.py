#start_of_parameters
#key=question1;  type=constant;  value=enter_question1
#key=question2;  type=constant;  value=enter_question2
#key=source_file;  type=constant;  value=enter_source_file
#key=return_column;  type=constant;  value=enter_return_column
#end_of_parameters
import pandas as pd
import numpy as np

source_file = "{source_file}"

if 'df' not in globals():
    df = pd.read_csv(workdir+source_file)
    dict1 = df.set_index('qid1')['question1'].fillna('none').to_dict()
    dict2 = df.set_index('qid2')['question2'].fillna('none').to_dict()
    dict_qid = {**dict1, **dict2}
    dict_quest = dict (zip(dict_qid.values(),dict_qid.keys()))
    nrows = len(df)
    next_qid = max(dict_qid.keys())+1

q1 = "{question1}"
q2 = "{question2}"

if q1 in dict_quest.keys():
    qid1 = dict_quest[q1]
else:
    qid1 = next_qid
    dict_quest[q1] = next_qid
    next_qid+=1

if q2 in dict_quest.keys():
    qid2 = dict_quest[q2]
else:
    qid2 = next_qid
    dict_quest[q2] = next_qid
    next_qid+=1

df_add = pd.DataFrame()
df_add.at[0, "id"] = nrows
df_add.at[0, "qid1"] = qid1
df_add.at[0, "qid2"] = qid2
df_add.at[0, "question1"] = q1
df_add.at[0, "question2"] = q2

pd.set_option('display.width', 500)

#print (df_add)

i = 0
for agent in agents:
    i+=1
    #print ("applying agent", i)
    agent.apply(df_add)
    if i==1:
        df_add = df_add.apply(pd.to_numeric)
    #print (df_add)

print(df_add.at[0,"{return_column}"])
