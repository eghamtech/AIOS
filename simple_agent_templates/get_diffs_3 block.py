import pandas as pd

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}

field_prefix = 'diffs3_'

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)

###########################################################################

from collections import Counter
from nltk.corpus import stopwords
import numpy as np



fldprefix = field_prefix + str(result_id)

    
def get_weight(count, eps=10000, min_count=2):
        return 0 if count < min_count else 1 / (count + eps)

train_qs = pd.Series(df[col1].tolist() + df[col2].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

stops = set(stopwords.words("english"))

def word_shares(row):
    q1_list = str(row[0]).lower().split()
    q1 = set(q1_list)
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0:0:0:0'

    q2_list = str(row[1]).lower().split()
    q2 = set(q2_list)
    q2words = q2.difference(stops)
    if len(q2words) == 0:
        return '0:0:0:0:0:0:0:0'

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q1_weights

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)

print (1)
    
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

print (2)

x = pd.DataFrame()

x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
x['word_match_2root'] = np.sqrt(x['word_match'])
x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
print (3)
x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
x['shared_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
x['cosine']           = df['word_shares'].apply(lambda x: float(x.split(':')[6]))
x['words_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))
x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']
print (4)
x['len_q1'] = df[col1].apply(lambda x: len(str(x)))
x['len_q2'] = df[col2].apply(lambda x: len(str(x)))
x['diff_len'] = x['len_q1'] - x['len_q2']
print (5)
x['caps_count_q1'] = df[col1].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
x['caps_count_q2'] = df[col2].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']
print (6)
x['len_char_q1'] = df[col1].apply(lambda x: len(str(x).replace(' ', '')))
x['len_char_q2'] = df[col2].apply(lambda x: len(str(x).replace(' ', '')))
x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']
print (7)
x['len_word_q1'] = df[col1].apply(lambda x: len(str(x).split()))
x['len_word_q2'] = df[col2].apply(lambda x: len(str(x).split()))
x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']
print (8)
x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']
print (9)
x['exactly_same'] = (df[col1] == df[col2]).astype(int)
x['duplicated'] = df.duplicated([col1,col2]).astype(int)

cnames = []
for i in range(1,len(x.columns)+1):
    col1 = fldprefix + '_' + str(i)
    cnames.append(col1)
x.columns = cnames


total_cols = 27

for i in range(1,total_cols+1):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    x[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname)

