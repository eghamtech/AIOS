if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    from collections import Counter
    from nltk.corpus import stopwords
    import numpy as np
        
    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    col_definition2 = "{random_dict}"
    col2 = col_definition2.split("|")[0]
    file2 = col_definition2.split("|")[1]
    result_id = {id}
    field_prefix = 'diffs3_'

    
    def __init__(self):
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        self.fldprefix = self.field_prefix + str(self.result_id)
        
        train_qs = self.pd.Series(self.df[self.col1].tolist() + self.df[self.col2].tolist()).astype(str)
        words = (" ".join(train_qs)).lower().split()
        counts = self.Counter(words)
        self.weights = {word: self.get_weight(count) for word, count in counts.items()}
        self.stops = set(self.stopwords.words("english"))

    def get_weight(self, count, eps=10000, min_count=2):
        return 0 if count < min_count else 1 / (count + eps)

    def word_shares(self, row):
        eps = 1e-5
        q1_list = str(row[0]).lower().split()
        q1 = set(q1_list)
        q1words = q1.difference(self.stops)
        if len(q1words) == 0:
            return '0:0:0:0:0:0:0:0'

        q2_list = str(row[1]).lower().split()
        q2 = set(q2_list)
        q2words = q2.difference(self.stops)
        if len(q2words) == 0:
            return '0:0:0:0:0:0:0:0'

        words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

        q1stops = q1.intersection(self.stops)
        q2stops = q2.intersection(self.stops)

        q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
        q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

        shared_2gram = q1_2gram.intersection(q2_2gram)

        shared_words = q1words.intersection(q2words)
        shared_weights = [self.weights.get(w, 0) for w in shared_words]
        q1_weights = [self.weights.get(w, 0) for w in q1words]
        q2_weights = [self.weights.get(w, 0) for w in q2words]
        total_weights = q1_weights + q2_weights

        R1 = self.np.sum(shared_weights) / (eps + self.np.sum(total_weights)) #tfidf share
        R2 = len(shared_words) / (eps + len(q1words) + len(q2words) - len(shared_words)) #count share
        R31 = len(q1stops) / (eps + len(q1words)) #stops in q1
        R32 = len(q2stops) / (eps + len(q2words)) #stops in q2
        Rcosine_denominator = (self.np.sqrt(self.np.dot(q1_weights,q1_weights))*self.np.sqrt(self.np.dot(q2_weights,q2_weights)))
        Rcosine = self.np.dot(shared_weights, shared_weights)/(eps + Rcosine_denominator)
        if len(q1_2gram) + len(q2_2gram) == 0:
            R2gram = 0
        else:
            R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
        return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)

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
        
        self.dfx['word_shares'] = self.dfx.apply(self.word_shares, axis=1, raw=True)


        df_run[self.fldprefix + '_1']       = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[0]))  #word_match
        df_run[self.fldprefix + '_2'] = self.np.sqrt(df_run[self.fldprefix + '_1'])  #word_match_2root
        df_run[self.fldprefix + '_3'] = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[1]))  #tfidf_word_match
        df_run[self.fldprefix + '_4']     = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[2]))  #shared_count
        
        df_run[self.fldprefix + '_5']     = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[3]))  #stops1_ratio
        df_run[self.fldprefix + '_6']     = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[4]))  #stops2_ratio
        df_run[self.fldprefix + '_7']     = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[5]))  #shared_2gram
        df_run[self.fldprefix + '_8']           = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[6]))  #cosine
        df_run[self.fldprefix + '_9']    = self.dfx['word_shares'].apply(lambda x: float(x.split(':')[7]))  #words_hamming
        df_run[self.fldprefix + '_10']     = df_run[self.fldprefix + '_5'] - df_run[self.fldprefix + '_6']  #diff_stops_r
        
        df_run[self.fldprefix + '_11'] = self.dfx[self.col1].apply(lambda x: len(str(x)))  #len_q1
        df_run[self.fldprefix + '_12'] = self.dfx[self.col2].apply(lambda x: len(str(x)))  #len_q2
        df_run[self.fldprefix + '_13'] = df_run[self.fldprefix + '_11'] - df_run[self.fldprefix + '_12']  #diff_len
        
        df_run[self.fldprefix + '_14'] = self.dfx[self.col1].apply(lambda x:sum(1 for i in str(x) if i.isupper()))  #caps_count_q1
        df_run[self.fldprefix + '_15'] = self.dfx[self.col2].apply(lambda x:sum(1 for i in str(x) if i.isupper()))  #caps_count_q2
        df_run[self.fldprefix + '_16'] = df_run[self.fldprefix + '_14'] - df_run[self.fldprefix + '_15']  #diff_caps
        
        df_run[self.fldprefix + '_17'] = self.dfx[self.col1].apply(lambda x: len(str(x).replace(' ', '')))  #len_char_q1
        df_run[self.fldprefix + '_18'] = self.dfx[self.col2].apply(lambda x: len(str(x).replace(' ', '')))  #len_char_q2
        df_run[self.fldprefix + '_19'] = df_run[self.fldprefix + '_17'] - df_run[self.fldprefix + '_18']  #diff_len_char
        
        df_run[self.fldprefix + '_20'] = self.dfx[self.col1].apply(lambda x: len(str(x).split()))  #len_word_q1
        df_run[self.fldprefix + '_21'] = self.dfx[self.col2].apply(lambda x: len(str(x).split()))  #len_word_q2
        df_run[self.fldprefix + '_22'] = df_run[self.fldprefix + '_20'] - df_run[self.fldprefix + '_21']  #diff_len_word
        
        df_run[self.fldprefix + '_23'] = df_run[self.fldprefix + '_17'] / df_run[self.fldprefix + '_20']  #avg_world_len1
        df_run[self.fldprefix + '_24'] = df_run[self.fldprefix + '_18'] / df_run[self.fldprefix + '_21']  #avg_world_len2
        df_run[self.fldprefix + '_25'] = df_run[self.fldprefix + '_23'] - df_run[self.fldprefix + '_24']  #diff_avg_word
        
        df_run[self.fldprefix + '_26'] = (self.dfx[self.col1] == self.dfx[self.col2]).astype(int)  #exactly_same
        df_run[self.fldprefix + '_27'] = self.dfx.duplicated([self.col1,self.col2]).astype(int)  #duplicated

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.run_on(self.df)
        
        total_cols = 27

        nrow = len(self.df)

        for i in range(1,total_cols+1):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))
        
    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
