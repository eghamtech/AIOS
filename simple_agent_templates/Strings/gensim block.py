#start_of_parameters
#key=google_news_file_path;  type=constant;  value=enter_google_news_file_path
#end_of_parameters

# possible value of google_news_file_path is ../GoogleNews-vectors-negative300.bin.gz

#no_permutation

if 'dicts' not in globals():
    dicts = {}

from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import gensim
    import numpy as np
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

    
    def func(self, s):
        return s[:1].lower() + s[1:] if s else ''
    
    def __init__(self):
        self.fldprefix = self.field_prefix + str(self.result_id)
        
        print ("loading models...")
        self.model = self.gensim.models.KeyedVectors.load_word2vec_format(workdir + "{google_news_file_path}", binary=True)
        print ("model1 loaded. next...")
        self.norm_model = self.gensim.models.KeyedVectors.load_word2vec_format(workdir + "{google_news_file_path}", binary=True)
        print ("norm_model loaded")
        self.norm_model.init_sims(replace=True)
        print ("ok")

    def wmd(self, s1, s2):
        s1 = s1.lower().split()
        s2 = s2.lower().split()
        s1 = [w for w in s1 if w not in self.stopwords]
        s2 = [w for w in s2 if w not in self.stopwords]
        return self.model.wmdistance(s1, s2)


    def norm_wmd(self, s1, s2):
        s1 = s1.lower().split()
        s2 = s2.lower().split()
        s1 = [w for w in s1 if w not in self.stopwords]
        s2 = [w for w in s2 if w not in self.stopwords]
        return self.norm_model.wmdistance(s1, s2)

    def sent2vec(self, words):
        words = [w for w in words if not w in self.stopwords]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(self.model[w])
            except:
                continue
        if len(M)==0:
            M = [[0.0]*300]
        M = self.np.array(M)
        v = M.sum(axis=0)
        return v / (1e-6+self.np.sqrt((v ** 2).sum()))

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
            s2v_a = self.sent2vec(wta)
            s2v_b = self.sent2vec(wtb)

            df_run.set_value(index, self.fldprefix + '_1', self.wmd(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_2', self.norm_wmd(sline1, sline2))
            df_run.set_value(index, self.fldprefix + '_3', cosine(s2v_a, s2v_b))
            df_run.set_value(index, self.fldprefix + '_4', cityblock(s2v_a, s2v_b))
            df_run.set_value(index, self.fldprefix + '_5', jaccard(s2v_a, s2v_b))
            df_run.set_value(index, self.fldprefix + '_6', canberra(s2v_a, s2v_b))
            df_run.set_value(index, self.fldprefix + '_7', euclidean(s2v_a, s2v_b))
            df_run.set_value(index, self.fldprefix + '_8', minkowski(s2v_a, s2v_b, 3))
            df_run.set_value(index, self.fldprefix + '_9', braycurtis(s2v_a, s2v_b))
            df_run.set_value(index, self.fldprefix + '_10', skew(s2v_a))
            df_run.set_value(index, self.fldprefix + '_11', skew(s2v_b))
            df_run.set_value(index, self.fldprefix + '_12', kurtosis(s2v_a))
            df_run.set_value(index, self.fldprefix + '_13', kurtosis(s2v_b))


            if i>=block and block>=1000:
                i=0
                print (index)

        df_run[[self.fldprefix + '_3',self.fldprefix + '_5',self.fldprefix + '_9']]=df_run[[self.fldprefix + '_3',self.fldprefix + '_5',self.fldprefix + '_9']].fillna(value=1.0)

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        
        self.run_on(self.df)
        
        nrow = len(self.df)

        for i in range(1,14):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()

