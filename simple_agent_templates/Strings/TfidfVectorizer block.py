import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk.corpus
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')
stopwords.append('would')

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}

field_prefix = 'tfidf_'

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)




func = lambda s: s[:1].lower() + s[1:] if s else ''



print ("creating TfidfVectorizer...")
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tfidf_txt = pd.Series(df[col1].tolist() + df[col2].tolist()).astype(str)
tfidf.fit_transform(tfidf_txt)
print ( "ok" )



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

    tf1 = tfidf.transform([sline1]).data
    tf2 = tfidf.transform([sline2]).data
    m1 = np.mean(tf1)
    m2 = np.mean(tf2)
    if math.isnan(m1):
        m1 = 1.0
    if math.isnan(m2):
        m2 = 1.0

    df.set_value(index, fldprefix + '_1', np.sum(tf1))
    df.set_value(index, fldprefix + '_2', np.sum(tf2))
    df.set_value(index, fldprefix + '_3', m1)
    df.set_value(index, fldprefix + '_4', m2)
    df.set_value(index, fldprefix + '_5', len(tf1))
    df.set_value(index, fldprefix + '_6', len(tf2))
    
    if i>=block:
        i=0
        print (index)

        
for i in range(1,7):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    df[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname)


