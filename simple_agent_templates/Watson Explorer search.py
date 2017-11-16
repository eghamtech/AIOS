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
output_column = field_prefix + str(result_id)
output_filename = output_column + ".csv"


count_of_words = 100

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
            try:
                dict_result[el] = ""
                links_done = []
                html = requests.get("http://172.19.112.116:9080/vivisimo/cgi-bin/query-meta.exe?v%3Asources=Wikipedia-Random-SC&v%3Aproject=Wikipedia&query=" + urllib.parse.quote_plus(el[:255])).text
                soup = BeautifulSoup(html, "lxml")
                for link in soup.findAll('a'):
                    url = link.get('href')
                    if (url[:5]=="http:" or url[:6]=="https:") and (url not in links_done):
                        links_done.append(url)
                        #print ("   load "+url)

                        html = requests.get(url).text
                        doc = fromstring(html)
                        tags = ['h1','h2','h3','h4','h5','h6', 'div', 'span', 'img', 'area', 'map']
                        args = {'meta':False, 'safe_attrs_only':False, 'page_structure':False, 'scripts':True, 'style':True, 'links':True, 'remove_tags':tags}
                        cleaner = Cleaner(**args)
                        path = '/html/body'
                        body = doc.xpath(path)[0]
                        result = cleaner.clean_html(body).text_content().encode('ascii', 'ignore')

                        dict_result[el] += "\n\n " + " ".join(str(result).split(" ")[:count_of_words])
            except:
                print("error at ", el[:100])
                dict_result[el] = ""
        else:
            dict_result[el] = ""

        idx+=1
        count+=1
        if idx>=block:
            idx=0
            print("processing item "+str(count)+" of "+str(ntotal))
            print ("work with: ", el[:100]+"...")


    def my_tokenize(text):
        filter1 = ',.;`-+#"!?:[]()%$=/\'*@~^&|\\'
        for ch in filter1:
            text = text.replace(ch, ' ')
        return text.strip().split()

    common_words = set([])
    for key in dict_result.keys():
        words = my_tokenize(dict_result[key])
        if len(common_words)==0:
            common_words = set(words)
        else:
            common_words = common_words.intersection(set(words))

    print ("common_words count: ", len(common_words))

    for key in dict_result.keys():
        words = set(my_tokenize(dict_result[key])) - common_words
        snew = ' '.join(words)
        dict_result[key] = snew


    df[output_column] = df['text_for_google'].map(dict_result)

    def make_dict(col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))

    dict_out = make_dict(df[output_column])

    if len(dict_out)<=1:
        print("result column is empty")
        print("#error")
    else:
        df[output_column] = df[output_column].map(dict_out)    #convert new column to dict
        pd.DataFrame(list(dict_out.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+output_column+'.csv')    #save new column dict
        df[[output_column]].to_csv(workdir+output_filename)
        print ("#add_field:"+output_column+",Y,"+output_filename)

