#no_permutation

if 'dicts' not in globals():
    dicts = {}

from nltk.tokenize import sent_tokenize, word_tokenize

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    from fuzzywuzzy import fuzz
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

    def func(self, s):
        return s[:1].lower() + s[1:] if s else ''
    
    def __init__(self):
        self.fldprefix = self.field_prefix + str(self.result_id)
    
    def get_wordnet_pos(self, pos_tag):
        if pos_tag[1].startswith('J'):
            return (pos_tag[0], self.wordnet.ADJ)
        elif pos_tag[1].startswith('V'):
            return (pos_tag[0], self.wordnet.VERB)
        elif pos_tag[1].startswith('N'):
            return (pos_tag[0], self.wordnet.NOUN)
        elif pos_tag[1].startswith('R'):
            return (pos_tag[0], self.wordnet.ADV)
        else:
            return (pos_tag[0], self.wordnet.NOUN)
    
    def run_on(self, df_run):
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        if self.col2 not in dicts:
            self.dict2 = self.pd.read_csv(workdir+'dict_'+self.col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict2 = {v:k for k,v in dicts[self.col2].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
        self.dfx[self.col2] = df_run[self.col2].map(self.dict2)

        block = int(len(df_run)/50)
        i = 0

        for index, row in self.dfx.iterrows():
            i+=1
            if type(row[self.col1])==str:
                sline1 = self.func(row[self.col1])
            else:
                sline1 = ''
            if type(row[self.col2])==str:
                sline2 = self.func(row[self.col2])
            else:
                sline2 = ''

            wta = word_tokenize(sline1.lower())
            wtb = word_tokenize(sline2.lower())

            tagset = None
            pos_a = map(self.get_wordnet_pos, self.nltk.tag._pos_tag(wta, tagset, self.tagger))
            pos_b = map(self.get_wordnet_pos, self.nltk.tag._pos_tag(wtb, tagset, self.tagger))

            tokens_a = [token.lower().strip(self.string.punctuation) for token in wta if token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_b = [token.lower().strip(self.string.punctuation) for token in wtb if token.lower().strip(self.string.punctuation) not in self.stopwords]
            stems_a = [self.stemmer.stem(token) for token in tokens_a]
            stems_b = [self.stemmer.stem(token) for token in tokens_b]
            lemma_a = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_a \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_b = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_b \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]


            sline_tok_a = ' '.join(tokens_a)
            sline_tok_b = ' '.join(tokens_b)
            sline_stem_a = ' '.join(stems_a)
            sline_stem_b = ' '.join(stems_b)
            sline_lemm_a = ' '.join(lemma_a)
            sline_lemm_b = ' '.join(lemma_b)

            df_run.set_value(index, self.fldprefix + '_1', self.fuzz.QRatio(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_2', self.fuzz.WRatio(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_3', self.fuzz.partial_ratio(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_4', self.fuzz.partial_token_set_ratio(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_5', self.fuzz.partial_token_sort_ratio(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_6', self.fuzz.token_set_ratio(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_7', self.fuzz.token_sort_ratio(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_8', self.fuzz.ratio(sline_tok_a, sline_tok_b))
            df_run.set_value(index, self.fldprefix + '_9', self.fuzz.partial_ratio(sline_tok_a, sline_tok_b))
            df_run.set_value(index, self.fldprefix + '_10', self.fuzz.token_sort_ratio(sline_tok_a, sline_tok_b))
            df_run.set_value(index, self.fldprefix + '_11', self.fuzz.token_set_ratio(sline_tok_a, sline_tok_b))
            df_run.set_value(index, self.fldprefix + '_12', self.fuzz.ratio(sline_stem_a, sline_stem_b))
            df_run.set_value(index, self.fldprefix + '_13', self.fuzz.partial_ratio(sline_stem_a, sline_stem_b))
            df_run.set_value(index, self.fldprefix + '_14', self.fuzz.token_sort_ratio(sline_stem_a, sline_stem_b))
            df_run.set_value(index, self.fldprefix + '_15', self.fuzz.token_set_ratio(sline_stem_a, sline_stem_b))
            df_run.set_value(index, self.fldprefix + '_16', self.fuzz.ratio(sline_lemm_a, sline_lemm_b))
            df_run.set_value(index, self.fldprefix + '_17', self.fuzz.partial_ratio(sline_lemm_a, sline_lemm_b))
            df_run.set_value(index, self.fldprefix + '_18', self.fuzz.token_sort_ratio(sline_lemm_a, sline_lemm_b))
            df_run.set_value(index, self.fldprefix + '_19', self.fuzz.token_set_ratio(sline_lemm_a, sline_lemm_b))

            if i>=block and block>=1000:
                i=0
                print (index)
    
    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        
        self.run_on(self.df)
        
        nrow = len(self.df)

        for i in range(1,20):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()

