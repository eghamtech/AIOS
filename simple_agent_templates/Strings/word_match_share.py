import pandas as pd
import nltk.corpus
import string
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

field_prefix = "field_"
output_filename = field_prefix + str(result_id) + ".csv"

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)


def word_match_share(s1, s2):
    q1words = {}
    q2words = {}
    for word in s1.split():
        if word not in stopwords:
            q1words[word] = 1
    for word in s2.split():
        if word not in stopwords:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

func = lambda s: s[:1].lower() + s[1:] if s else ''



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
    
    df.set_value(index, fldprefix + '_1', word_match_share(sline1, sline2))
    
    if i>=block:
        i=0
        print (index)

        
        
newfields = []

for i in range(1,2):
    newfields.append(fldprefix + '_' + str(i))
    
df[newfields].to_csv(workdir+output_filename)

for fld in newfields:
    print ("#add_field:"+fld+",N,"+output_filename)

