#no_permutation

if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
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

    def func(self, s):
        return s[:1].lower() + s[1:] if s else ''
    
    def __init__(self):
        self.fldprefix = self.field_prefix + str(self.result_id)

    def word_match_share(self, s1, s2):
        q1words = {}
        q2words = {}
        for word in s1.split():
            if word not in self.stopwords:
                q1words[word] = 1
        for word in s2.split():
            if word not in self.stopwords:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R
        
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
        
            df_run.set_value(index, self.fldprefix + '_1', self.word_match_share(sline1, sline2))

            if i>=block and block>=1000:
                i=0
                print (index)
                
    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        
        self.run_on(self.df)
        
        newfields = []

        for i in range(1,2):
            newfields.append(self.fldprefix + '_' + str(i))

        self.df[newfields].to_csv(workdir+self.output_filename)

        nrow = len(self.df)

        for fld in newfields:
            print ("#add_field:"+fld+",N,"+self.output_filename+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
