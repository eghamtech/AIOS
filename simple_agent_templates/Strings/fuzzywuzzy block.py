import pandas as pd
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import nltk.corpus
import nltk.stem.snowball
stemmer = nltk.stem.snowball.SnowballStemmer('english')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

from nltk.corpus import wordnet
from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

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

field_prefix = 'fuzz_'

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)


func = lambda s: s[:1].lower() + s[1:] if s else ''

def get_wordnet_pos(pos_tag):
    if pos_tag[1].startswith('J'):
        return (pos_tag[0], wordnet.ADJ)
    elif pos_tag[1].startswith('V'):
        return (pos_tag[0], wordnet.VERB)
    elif pos_tag[1].startswith('N'):
        return (pos_tag[0], wordnet.NOUN)
    elif pos_tag[1].startswith('R'):
        return (pos_tag[0], wordnet.ADV)
    else:
        return (pos_tag[0], wordnet.NOUN)

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
    
    tagset = None
    pos_a = map(get_wordnet_pos, nltk.tag._pos_tag(wta, tagset, tagger))
    pos_b = map(get_wordnet_pos, nltk.tag._pos_tag(wtb, tagset, tagger))

    tokens_a = [token.lower().strip(string.punctuation) for token in wta if token.lower().strip(string.punctuation) not in stopwords]
    tokens_b = [token.lower().strip(string.punctuation) for token in wtb if token.lower().strip(string.punctuation) not in stopwords]
    stems_a = [stemmer.stem(token) for token in tokens_a]
    stems_b = [stemmer.stem(token) for token in tokens_b]
    lemma_a = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_a \
                if token.lower().strip(string.punctuation) not in stopwords]
    lemma_b = [lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_b \
                if token.lower().strip(string.punctuation) not in stopwords]

    
    sline_tok_a = ' '.join(tokens_a)
    sline_tok_b = ' '.join(tokens_b)
    sline_stem_a = ' '.join(stems_a)
    sline_stem_b = ' '.join(stems_b)
    sline_lemm_a = ' '.join(lemma_a)
    sline_lemm_b = ' '.join(lemma_b)

    df.set_value(index, fldprefix + '_1', fuzz.QRatio(sline1, sline2))
    df.set_value(index, fldprefix + '_2', fuzz.WRatio(sline1, sline2))
    df.set_value(index, fldprefix + '_3', fuzz.partial_ratio(sline1, sline2))
    df.set_value(index, fldprefix + '_4', fuzz.partial_token_set_ratio(sline1, sline2))
    df.set_value(index, fldprefix + '_5', fuzz.partial_token_sort_ratio(sline1, sline2))
    df.set_value(index, fldprefix + '_6', fuzz.token_set_ratio(sline1, sline2))
    df.set_value(index, fldprefix + '_7', fuzz.token_sort_ratio(sline1, sline2))
    df.set_value(index, fldprefix + '_8', fuzz.ratio(sline_tok_a, sline_tok_b))
    df.set_value(index, fldprefix + '_9', fuzz.partial_ratio(sline_tok_a, sline_tok_b))
    df.set_value(index, fldprefix + '_10', fuzz.token_sort_ratio(sline_tok_a, sline_tok_b))
    df.set_value(index, fldprefix + '_11', fuzz.token_set_ratio(sline_tok_a, sline_tok_b))
    df.set_value(index, fldprefix + '_12', fuzz.ratio(sline_stem_a, sline_stem_b))
    df.set_value(index, fldprefix + '_13', fuzz.partial_ratio(sline_stem_a, sline_stem_b))
    df.set_value(index, fldprefix + '_14', fuzz.token_sort_ratio(sline_stem_a, sline_stem_b))
    df.set_value(index, fldprefix + '_15', fuzz.token_set_ratio(sline_stem_a, sline_stem_b))
    df.set_value(index, fldprefix + '_16', fuzz.ratio(sline_lemm_a, sline_lemm_b))
    df.set_value(index, fldprefix + '_17', fuzz.partial_ratio(sline_lemm_a, sline_lemm_b))
    df.set_value(index, fldprefix + '_18', fuzz.token_sort_ratio(sline_lemm_a, sline_lemm_b))
    df.set_value(index, fldprefix + '_19', fuzz.token_set_ratio(sline_lemm_a, sline_lemm_b))

    if i>=block:
        i=0
        print (index)

nrow = len(df)

for i in range(1,20):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    df[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


