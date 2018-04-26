#start_of_parameters
#key=URL;  type=constant;  value=enter_URL
#key=username;  type=constant;  value=enter_username
#key=password;  type=constant;  value=enter_password
#key=system_id;  type=constant;  value=enter_system_id
#key=layout_listing_id;  type=constant;  value=enter_layout_listing_id
#key=search_id;  type=constant;  value=enter_search_id
#key=return_column;  type=constant;  value=enter_return_column
#end_of_parameters
import requests
import json
import base64
import re
import pandas as pd

URL = '{URL}'
system_id = {system_id}
layout_listing_id = {layout_listing_id}

client = requests.session()
client.get(URL, verify=False)
csrf = client.cookies['Csrf-token']
token2 = base64.b64encode(b"{username}:{password}")
headers = {'Accept':'application/json','X-Csrf-Token':csrf, 'Authorization':'Basic '+token2.decode("ascii")}

r=requests.get(URL + "/system-" + str(system_id) + "/config-jail/api/v1/layout_listings/config/" + str(layout_listing_id), headers=headers, verify=False)
jsconfig = json.loads(r.text)

r=requests.get(URL + "/system-" + str(system_id) + "/config-jail/api/v1/applications?layout=" + str(layout_listing_id) + "&saved_search=" + str(search_id), headers=headers, verify=False)
jsdata = json.loads(r.text)

jsout = {"training_data": []}

for obj in jsdata:
    obj1 = {}
    for i in range(0, len(jsconfig["columns"])):
        conf = jsconfig["columns"][i]
        heading = conf["heading"]
        if not (conf["instance_string"] is None):
            heading = heading + " (" + conf["instance_string"] + ")"
        if conf["data_type"]=="LOOKUP":
            obj1[heading] = obj["layout_data"][i]["value_raw"]
        else:
            obj1[heading] = obj["layout_data"][i]["value"]
    jsout["training_data"].append(obj1)
    
df_add = pd.DataFrame().from_dict(jsout["training_data"])
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