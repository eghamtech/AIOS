#start_of_parameters
#key=google_news_file_path;  type=constant;  value=../GoogleNews-vectors-negative300.bin.gz
#end_of_parameters

import pandas as pd
import gensim
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import numpy as np
from nltk.tokenize import word_tokenize
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

field_prefix = 'gensim_'

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)



def wmd(s1, s2):
    s1 = s1.lower().split()
    s2 = s2.lower().split()
    s1 = [w for w in s1 if w not in stopwords]
    s2 = [w for w in s2 if w not in stopwords]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = s1.lower().split()
    s2 = s2.lower().split()
    s1 = [w for w in s1 if w not in stopwords]
    s2 = [w for w in s2 if w not in stopwords]
    return norm_model.wmdistance(s1, s2)

def sent2vec(words):
    words = [w for w in words if not w in stopwords]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    if len(M)==0:
        M = [[0.0]*300]
    M = np.array(M)
    v = M.sum(axis=0)
    return v / (1e-6+np.sqrt((v ** 2).sum()))


func = lambda s: s[:1].lower() + s[1:] if s else ''


print ("loading models...")
model = gensim.models.KeyedVectors.load_word2vec_format(workdir + "{google_news_file_path}", binary=True)
print ("model1 loaded. next...")
norm_model = gensim.models.KeyedVectors.load_word2vec_format(workdir + "{google_news_file_path}", binary=True)
print ("norm_model loaded")
norm_model.init_sims(replace=True)
print ("ok")



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
    
    wta = word_tokenize(sline1.lower())
    wtb = word_tokenize(sline2.lower())
    s2v_a = sent2vec(wta)
    s2v_b = sent2vec(wtb)
    
    df.set_value(index, fldprefix + '_1', wmd(sline1, sline2))
    df.set_value(index, fldprefix + '_2', norm_wmd(sline1, sline2))
    df.set_value(index, fldprefix + '_3', cosine(s2v_a, s2v_b))
    df.set_value(index, fldprefix + '_4', cityblock(s2v_a, s2v_b))
    df.set_value(index, fldprefix + '_5', jaccard(s2v_a, s2v_b))
    df.set_value(index, fldprefix + '_6', canberra(s2v_a, s2v_b))
    df.set_value(index, fldprefix + '_7', euclidean(s2v_a, s2v_b))
    df.set_value(index, fldprefix + '_8', minkowski(s2v_a, s2v_b, 3))
    df.set_value(index, fldprefix + '_9', braycurtis(s2v_a, s2v_b))
    df.set_value(index, fldprefix + '_10', skew(s2v_a))
    df.set_value(index, fldprefix + '_11', skew(s2v_b))
    df.set_value(index, fldprefix + '_12', kurtosis(s2v_a))
    df.set_value(index, fldprefix + '_13', kurtosis(s2v_b))
    
    
    if i>=block:
        i=0
        print (index)

df[[fldprefix + '_3',fldprefix + '_5',fldprefix + '_9']]=df[[fldprefix + '_3',fldprefix + '_5',fldprefix + '_9']].fillna(value=1.0)

nrow = len(df)

for i in range(1,14):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    df[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


