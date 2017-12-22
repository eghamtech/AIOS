if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
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

    def func(self, s):
        return s[:1].lower() + s[1:] if s else ''
    
    def __init__(self):
        self.fldprefix = self.field_prefix + str(self.result_id)
        
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        if self.col2 not in dicts:
            self.dict2 = self.pd.read_csv(workdir+'dict_'+self.col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict2 = {v:k for k,v in dicts[self.col2].items()} # make key=number, value=string
        
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = self.df[self.col1].map(self.dict1)
        self.dfx[self.col2] = self.df[self.col2].map(self.dict2)
        
        print ("creating TfidfVectorizer...")
        self.tfidf = self.TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf_txt = self.pd.Series(self.dfx[self.col1].tolist() + self.dfx[self.col2].tolist()).astype(str)
        self.tfidf.fit_transform(tfidf_txt)
        print ( "ok" )

    def run_on(self, df_run):
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

            tf1 = self.tfidf.transform([sline1]).data
            tf2 = self.tfidf.transform([sline2]).data
            m1 = self.np.mean(tf1)
            m2 = self.np.mean(tf2)
            if self.math.isnan(m1):
                m1 = 1.0
            if self.math.isnan(m2):
                m2 = 1.0

            df_run.set_value(index, self.fldprefix + '_1', self.np.sum(tf1))
            df_run.set_value(index, self.fldprefix + '_2', self.np.sum(tf2))
            df_run.set_value(index, self.fldprefix + '_3', m1)
            df_run.set_value(index, self.fldprefix + '_4', m2)
            df_run.set_value(index, self.fldprefix + '_5', len(tf1))
            df_run.set_value(index, self.fldprefix + '_6', len(tf2))

            if i>=block and block>=1000:
                i=0
                print (index)
        
    def run(self, mode):
        print ("enter run mode " + str(mode))
                
        self.run_on(self.df)
        
        nrow = len(self.df)
        
        for i in range(1,7):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))
        
    def apply(self, df_add):
        
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()






