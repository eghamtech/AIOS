#start_of_parameters
#key=word_count_max;  type=constant;  value=0
#key=group_length;  type=constant;  value=300
#key=glove_host;  type=constant;  value=enter_glove_host
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns as elements of GloVe vector by parsing text field into words including punctuation
# it queries GloVe database on glove_host provided via AIOS API
# it also maps original 300 elements vector to group_length and considers only word_count_max initial words
# if parameter word_count_max not specified (0) then agent will analyse given field to find its maximum length

if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import re

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'glv_' + str({group_length}) + '_' + col1 + '_'
    fldprefix = field_prefix + str(result_id)
    nwords = {word_count_max}
    group_length = {group_length}
    numbers_count = nwords * int(300/group_length)
    error = 0
    max_words = 0
    
    def __init__(self):
        global dicts
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        
        # if parameter word_count_max == 0 then use max_words found in the given field
        if self.nwords == 0:
            # map dict fields
            dfx = self.pd.DataFrame()
            dfx[self.col1] = self.df[self.col1].map(self.dict1)
            # find size of longest in terms of words record
            self.max_words = dfx[self.col1].apply(self._no_of_words).max()
            self.nwords = self.max_words
            self.numbers_count = self.nwords * int(300/self.group_length)
            print("Longest record has " + str(self.max_words) + " words. Using it as GloVe size limit.")
            
        
    def _removeNonAscii(self, s): return "".join(i for i in s if ord(i)<128)
    
    # splits string into words including punctuation
    def _tokenize(self, s):
        swords = ''
        if type(s)==str:
           swords = ' '.join(self.re.findall(r"[\w'`]+|[.,!?;]", s))
        
        return swords 
    
    # counts number of words in a string
    def _no_of_words(self,s):
        _words = self._tokenize(s)
        return len(_words.split())
    
    def run_on(self, df_run):
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
        
        block = int(len(df_run)/500)
        i = 0

        import requests
        import json
        
        for index, row in self.dfx.iterrows():
            i+=1
            sline1 = self._tokenize(row[self.col1])
            
            #values = [0]*(self.nwords*300)
            r = requests.post("{glove_host}", verify=False, data={'action': 'glove_numbers', 'word_count_max': self.nwords, 'group_length': self.group_length, 'string': sline1})
            if r.status_code!=200:
                print(r.reason)
                print("#error")
                self.error = 1
                break
            
            obj = json.loads(r.text)
            values = [float(v) for v in obj['data'].split(',')]
            if len(values)!=self.numbers_count:
                print("wrong response length. got", len(values), ", must be", self.numbers_count, ", nwords", self.nwords, ", grp_length", self.group_length)
                print("string:", sline1)
                print("#error")
                self.error = 1
                break
            
            #if index==0 and len(df_run)>1:
            #    print('creating columns: ')
            #    last = 0
            cnt = len(values)
            for j in range(0, cnt):
                #if index==0 and len(df_run)>1:
                #    last+=1
                #    if last>=10:
                #        print(str(j) + "/" + str(cnt) + "...")
                #        last = 0
                df_run.set_value(index, self.fldprefix + '_' + str(j), values[j])

            if i>=block and block>=10:
                i=0
                print (index, sline1, values[0])
    
    def run(self, mode):
        print ("enter run mode " + str(mode))
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
            return
        
        cols = []
        # prepare dataframe with numbers_count new columns and init with 0.0 
        for i in range(0,self.numbers_count):
            fld = self.fldprefix + '_' + str(i)
            cols.append(fld)
        dfx2 = self.pd.DataFrame(0.0, index=self.np.arange(len(self.df)), columns=cols)
        
        print ("start adding columns")
        # join original column with new columns
        self.df = self.df.join(dfx2)
        print ("ended adding columns")
        
        self.run_on(self.df)
        
        if self.error==1:
            return
        
        nrow = len(self.df)
        
        # register and save new columns one by one
        for i in range(0,self.numbers_count):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        for i in range(0,self.numbers_count):
            fld = self.fldprefix + '_' + str(i)
            df_add[fld] = 0.0
        self.run_on(df_add)
    
agent_{id} = cls_agent_{id}()
