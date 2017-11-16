import pandas as pd
import numpy as np
import urllib.parse
from lxml.html import fromstring
from lxml.html.clean import Cleaner
import requests
from bs4 import BeautifulSoup

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]

result_id = {id}
output_filename = vr_file_prefix + str(result_id) + ".csv"

from watson_developer_cloud import VisualRecognitionV3 as VisualRecognitionV3

visual_recognition = VisualRecognitionV3(vr_api_date, api_key=vr_key )

df = pd.read_csv(workdir+file1)[[col1]]

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df["text_for_google"] = df[col1].map(dict1) #.fillna('')

count_non_empty = 0
for el in df.loc[df['text_for_google'].isnull()==False, 'text_for_google'].unique():
    if len(el)>0:
        count_non_empty +=1

if count_non_empty<=1:
    print("source column has less than 2 unique values")
    print("#error")
else:
    dict_result = {}

    block = int(len(df)/500)
    idx=block-1
    count=0
    ntotal = len(df.loc[df['text_for_google'].isnull()==False, 'text_for_google'].unique())
    for el in df.loc[df['text_for_google'].isnull()==False, 'text_for_google'].unique():
        #print ("work with: ", el[:100]+"...")
        if len(el)>0:
            dict_result[el] = {}
            links_done = []
            html = requests.get("http://172.19.112.116:9080/vivisimo/cgi-bin/query-meta.exe?v%3Asources=Wikipedia-Random-SC&v%3Aproject=Wikipedia&query=" + urllib.parse.quote_plus(el[:255])).text
            soup = BeautifulSoup(html, "lxml")
            for link in soup.findAll('a'):
                url = link.get('href')

                if (url[:5]=="http:" or url[:6]=="https:") and (url not in links_done):
                    links_done.append(url)
                    #print ("   load "+url)

                    html2 = requests.get(url).text

                    soup2 = BeautifulSoup(html2, "lxml")
                    count_found = 0
                    for link2 in soup2.findAll('img'):
                        url2 = link2.get('src')
                        #print (url2)
                        if (url2[:2]=="//" or url2[:5]=="http:" or url2[:6]=="https:") and (url2[-3:]=="jpg"): # or url2[-3:]=="png"):
                            #print (url2)
                            if url2[:2]=="//":
                                url2 = "http:" + url2
                            response = visual_recognition.classify(images_url=url2)

                            if "classifiers" in response["images"][0]:
                                aclasses = response["images"][0]["classifiers"][0]["classes"]
                                for col in aclasses:
                                    if col["class"] in dict_result[el].keys():
                                        dict_result[el][col["class"]] = max(dict_result[el][col["class"]], col["score"])
                                    else:
                                        dict_result[el][col["class"]] = col["score"]

                                count_found+=1
                                if count_found>=5:
                                    break

                #print (dict_result[el].keys(), "\n")
        else:
            dict_result[el] = {}

        idx+=1
        count+=1
        if idx>=block:
            idx=0
            print("processing item "+str(count)+" of "+str(ntotal))
            print ("work with: ", el[:100]+"...")


    dict_keywords = {}
    for key in dict_result.keys():
        col = dict_result[key]

        for item in col.keys():
            class1 = item
            score1 = col[item]
            if class1 in dict_keywords.keys():
                dict_keywords[class1] += 1
            else:
                dict_keywords[class1] = 1

    import operator
    sorted_keys = sorted(dict_keywords.items(), key=operator.itemgetter(1), reverse=True)
    print (len(sorted_keys))
    sorted_keys = sorted_keys[:50]
    print ("use " + str(len(sorted_keys)) + " keywords")

    if len(sorted_keys)==0:
        print("found zero classes")
        print("#error")
    else:
        field_idx = 0
        for skey in sorted_keys:
            field_idx += 1
            df['vr'+str(result_id)+'_'+str(field_idx)] = 0

        idx=-1
        for index, row in df.iterrows():
            key = row['text_for_google']
            if pd.isnull(key)==False:
                result = dict_result[key]
                nvalues = 0
                field_idx = 0
                for skey in sorted_keys:
                    field_idx += 1
                    skey1 = skey[0]
                    value = 0
                    for item in result.keys():
                        if item==skey1:
                            value = result[item]
                            nvalues+=1

                    if value!=0:
                        df.loc[index, 'vr'+str(result_id)+'_'+str(field_idx)] = value

            idx+=1
            if idx>=1000:
                print ("filling values for row " + str(index))
                idx=0

        print ("writing file " + output_filename)
        df.loc[:,'vr'+str(result_id)+'_1':].to_csv(workdir+output_filename)

        field_idx = 0
        for skey in sorted_keys:
            field_idx += 1
            print ("#add_field:vr"+str(result_id)+'_'+str(field_idx)+",N,"+output_filename)

