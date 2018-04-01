#start_of_parameters
#key=input_json;  type=constant;  value=enter_input_json_file
#key=output_json;  type=constant;  value=enter_output_json_file
#key=output_primary_field;  type=constant;  value=enter_output_primary_field
#key=return_column;  type=constant;  value=enter_return_column
#end_of_parameters

# Processes JSON file which has "training_data" object according to below specification in Wiki:
# https://github.com/eghamtech/AIOS/wiki/Input-data-JSON-format-01
# creates output_json file with only two fields: 
# [ {output_primary_field: value, return_column: value}, {..} .... {..} ]

import re
import pandas as pd
import json

#json_data = json.loads("""{input_json}""")
with open("{input_json}", encoding='utf-8') as f1:
   json_data = json.load(f1)

df_add = pd.DataFrame().from_dict(json_data["training_data"])
#print(df_add)
new_cols = []
# rename columns by removing symbols - use the same algorithm as used for training
for c in df_add.columns:
    str1 = c
    #str1 = re.sub('[^0-9a-zA-Z]+', '_', str1)
    for ch in [".", ",", " ", "/", "(", ")", "?", "!"]:
                str1 = str1.replace(ch, "_")
    new_cols.append(str1)

df_add.columns = new_cols

#print(df_add)

i = 0
# apply each required agent on df_add which will be extended by each agent until it reaches return_column
for agent in agents:
    i+=1
    #print ("applying agent", i, str(agent))
    agent.apply(df_add)
    #print (df_add)
    if i==1:
        df_add = df_add.apply(pd.to_numeric, errors='coerse')

print (','.join(str(x) for x in df_add["{return_column}"].values))

out_json = df_add[["{output_primary_field}","{return_column}"]].to_json(orient='records')
with open("{output_json}", 'w', encoding='utf-8') as fo:
            fo.write(out_json)

