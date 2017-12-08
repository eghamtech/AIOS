#start_of_parameters
#key=nlu_version;  type=constant;  value=enter_nlu_version
#key=nlu_username;  type=constant;  value=enter_nlu_username
#key=nlu_password;  type=constant;  value=enter_nlu_password
#end_of_parameters

import pandas as pd
import numpy as np

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]

result_id = {id}
output_filename = "nlu_" + str(result_id) + ".csv"

df = pd.read_csv(workdir+file1)[[col1]]

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df["text_for_nlu"] = df[col1].map(dict1).fillna('')

import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 as Features


natural_language_understanding = NaturalLanguageUnderstandingV1(version = "{nlu_version}", username = "{nlu_username}", password = "{nlu_password}")


dict_result = {}
dict_keywords = {}

block = int(len(df)/500)
idx=block-1
count=0
ntotal = len(df['text_for_nlu'].unique())
for el in df['text_for_nlu'].unique():
    if len(el)>0:
        try:
            response = natural_language_understanding.analyze(text = el,features=[Features.Keywords()],language='en')
            dict_result[el] = response["keywords"]
            for item in response["keywords"]:
                keyword = item["text"]
                relevance = item["relevance"]
                if keyword in dict_keywords.keys():
                    dict_keywords[keyword] += 1
                else:
                    dict_keywords[keyword] = 1
        except:
            print("error at ", el)
            dict_result[el] =[]
    else:
        dict_result[el] =[]

    idx+=1
    count+=1
    if idx>=block:
        idx=0
        print("NLU processing item "+str(count)+" of "+str(ntotal))

import operator
sorted_keys = sorted(dict_keywords.items(), key=operator.itemgetter(1), reverse=True)
print (len(sorted_keys))
sorted_keys = sorted_keys[:50]
print ("use " + str(len(sorted_keys)) + " keywords")

field_idx = 0
for skey in sorted_keys:
    field_idx += 1
    df['nlu'+str(result_id)+'_'+str(field_idx)] = 0

idx=-1
for index, row in df.iterrows():
    key = row['text_for_nlu']
    result = dict_result[key]
    nvalues = 0
    field_idx = 0
    for skey in sorted_keys:
        field_idx += 1
        skey1 = skey[0]
        value = 0
        for item in result:
            if item['text']==skey1:
                value = item['relevance']
                nvalues+=1
        
        if value!=0:
            df.loc[index, 'nlu'+str(result_id)+'_'+str(field_idx)] = value
    
    idx+=1
    if idx>=1000:
        print ("filling values for row " + str(index))
        idx=0

print ("writing file " + output_filename)
df.loc[:,'nlu'+str(result_id)+'_1':].to_csv(workdir+output_filename)

field_idx = 0
nrow = len(df)
for skey in sorted_keys:
    field_idx += 1
    print ("#add_field:nlu"+str(result_id)+'_'+str(field_idx)+",N,"+output_filename+","+str(nrow))
