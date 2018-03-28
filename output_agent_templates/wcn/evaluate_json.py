#start_of_parameters
#key=input_json;  type=constant;  value=enter_input_json
#key=return_column;  type=constant;  value=enter_return_column
#end_of_parameters
import re
import pandas as pd
import json

json_data = json.loads("""{input_json}""")

df_add = pd.DataFrame().from_dict(json_data["training_data"])
new_cols = []
for c in df_add.columns:
    str1 = c
    str1 = re.sub('[^0-9a-zA-Z]+', '_', str1)
    new_cols.append(str1)

df_add.columns = new_cols

#print(df_add)

i = 0
for agent in agents:
    i+=1
    #print ("applying agent", i)
    agent.apply(df_add)
    #print (df_add)
    if i==1:
        df_add = df_add.apply(pd.to_numeric, errors='coerse')

print (','.join(str(x) for x in df_add["{return_column}"].values))
