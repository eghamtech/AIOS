#start_of_parameters
#key=pers_insights_username;  type=constant;  value=enter_pers_insights_username
#key=pers_insights_password;  type=constant;  value=enter_pers_insights_password
#end_of_parameters

import pandas as pd
import numpy as np

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]

result_id = {id}
pers_insights_file_prefix = "pi_"
output_filename = pers_insights_file_prefix + str(result_id) + ".csv"

df = pd.read_csv(workdir+file1)[[col1]]

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

col_name = "text_for_pi"
df[col_name] = df[col1].map(dict1).fillna('')

########################################################
from watson_developer_cloud import PersonalityInsightsV3 as PersonalityInsightsV3
personality_insights = PersonalityInsightsV3(username= "{pers_insights_username}", password= "{pers_insights_password}")


dict_result = {}
dict_keywords = {}

block = int(len(df)/500)
idx=block-1
count=0

ntotal = len(df[col_name].unique())
for el in df[col_name].unique():
    if type(el)!=str:
        el = ''
    if len(el)>0:
        sline = el
        
        while len(sline)<10000:
            sline+=" "+el
        
        s={"contentItems": []}
    
        i1 = {"content": sline, "contenttype": "text/plain", "created": 1447639154000, "id": 1, "language": "en" }
        s["contentItems"].append(i1)
        
        profile = personality_insights.profile(s, content_type='application/json', raw_scores=True, consumption_preferences=False)
        aresult = []
        
        for key in ["personality", "needs", "values", "behavior"]:
            if key in profile:
                for item in profile[key]:
                    if key=="behavior":
                        aresult.append({"text":item["trait_id"], "percentile":item["percentage"]})
                        #print (item["trait_id"], item["percentage"])
                    else:
                        aresult.append({"text":item["trait_id"], "percentile":item["percentile"], "score":item["raw_score"]})
                        #print (item["trait_id"], item["percentile"], item["raw_score"])
                    if "children" in item:
                        for item2 in item["children"]:
                            aresult.append({"text":item2["trait_id"], "percentile":item2["percentile"], "score":item2["raw_score"]})
                            #print (item2["trait_id"], item2["percentile"], item2["raw_score"])
        
        dict_result[el] = aresult
        for item in aresult:
            keyword = item["text"]
            if keyword in dict_keywords.keys():
                dict_keywords[keyword] += 1
            else:
                dict_keywords[keyword] = 1
    else:
        dict_result[el] = []

    idx+=1
    count+=1
    if idx>=block:
        idx=0
        print("Perconality Insights processing item "+str(count)+" of "+str(ntotal))
        
        
import operator
sorted_keys = sorted(dict_keywords.items(), key=operator.itemgetter(1), reverse=True)
print (len(sorted_keys))
#sorted_keys = sorted_keys[:50]
print ("use " + str(len(sorted_keys)) + " keywords")

idx=block-1
cols = []
for index, row in df.iterrows():
    key = row[col_name]
    if type(key)!=str:
        key = ''
    result = dict_result[key]
    field_idx = 0
    for skey in sorted_keys:
        field_idx += 1
        skey1 = skey[0]
        value1 = -1
        value2 = -1
        for item in result:
            if item['text']==skey1:
                value1 = item['percentile']
                if "score" in item:
                    value2 = item["score"]

        
        if value1>=0:
            cname = 'pi_p_'+str(result_id)+'_'+str(field_idx)
            df.loc[index, cname] = value1
            if not (cname in cols):
                cols.append(cname)
        if value2>=0:
            cname = 'pi_s_'+str(result_id)+'_'+str(field_idx)
            df.loc[index, cname] = value2
            if not (cname in cols):
                cols.append(cname)
    
    idx+=1
    if idx>=block:
        print ("filling values for row " + str(index))
        idx=0

print ("writing file " + output_filename)

df[cols].to_csv(workdir+output_filename)

nrow = len(df)
for col1 in cols:
    print ("#add_field:"+col1+",N,"+output_filename+","+str(nrow))
