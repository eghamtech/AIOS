#start_of_parameters
#key=word_count_max;  type=constant;  value=0
#key=group_length;  type=constant;  value=1
#key=glove_host;  type=constant;  value=enter_glove_host
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns as elements of GloVe vector by parsing text field into words including punctuation
# it queries GloVe database on glove_host provided via AIOS API
# it also maps original 300 elements vector to "vector size"=300/group_length and considers only word_count_max initial words
# if parameter word_count_max not specified (0) then agent will consider all given words in each row
#
# this version is optimised for speed and memory usage and it saves temp file with results in case agent crashes
# this version produces single vector of size "vector size" which is a simple sum of all words vectors

if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import re
    import os

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'glv_sum_' + str({group_length}) + '_' + col1 + '_'
    temp_file_name = field_prefix + '.tmp'
    fldprefix = field_prefix + str(result_id)
    nwords = {word_count_max}
    group_length = {group_length}
    numbers_count = int(300/group_length)
    error = 0
    
    def __init__(self):
        global dicts
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
                  
        self.cols = []
        # prepare list of new columns
        for i in range(0,self.numbers_count):
            fld = self.fldprefix + '_' + str(i)
            self.cols.append(fld)
            
        # if saved temp object exists then load it from filesystem to carry on from last good batch
        if self.os.path.isfile(workdir + self.temp_file_name):
            self.df_np = self.pd.read_pickle(workdir + self.temp_file_name, compression='bz2')
            self.df_np = self.df_np.values.tolist()
            self.index_start_from = len(self.df_np)
            print ('df_np array loaded from temp file, continue conversion from row: ', self.index_start_from+1)
        else:
            self.df_np = []
            self.index_start_from = 0
        
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
        
        block = int(len(df_run)/20)
        i = 0
        
        import requests
        import json
        
        for index, row in self.dfx.iloc[self.index_start_from:].iterrows():
            i+=1
            sline1 = self._tokenize(row[self.col1])
            
            if len(sline1.split()) > 0:
                # if parameter word_count_max == 0 then use number of words in the current line
                if self.nwords == 0:
                    self.nwords = len(sline1.split())

                r = requests.post("{glove_host}", verify=False, data={'action': 'glove_numbers', 'word_count_max': self.nwords, 'group_length': self.group_length, 'string': sline1})
                if r.status_code!=200:
                    print(r.reason)
                    print("#error")
                    self.error = 1
                    break

                obj = json.loads(r.text)
                values = [self.np.float32(v) for v in obj['data'].split(',')]
                if len(values) != self.nwords*self.numbers_count:
                    print("wrong response length. got", len(values), ", must be", self.nwords*self.numbers_count, ", nwords", self.nwords, ", grp_length", self.group_length)
                    print("string:", sline1)
                    print("#error")
                    self.error = 1
                    break

                glove_array = self.np.array(values)                     # convert continous list of all words gloves to 1-D array
                glove_array = glove_array.reshape((self.nwords, -1))    # convert 1-D array to NWords*GloveSize array
                glove_array = glove_array.sum(axis=0)                   # simple sum of all words glove numbers
                values = glove_array.tolist()                           # convert back to list for faster appending
            else:
                values = [0] * self.numbers_count
               
            self.df_np.append(values)

            if i>=block and block>=10:
                i=0
                print (index, sline1, values[0])
                df_np_df = self.pd.DataFrame(self.df_np)
                df_np_df.to_pickle(workdir + self.temp_file_name, compression='bz2')
                del df_np_df
                print ('df_np array saved to temp file; length: ', len(self.df_np))
    
        self.df_np = self.pd.DataFrame(self.df_np, columns = self.cols)
        
        
    def run(self, mode):
        print ("enter run mode " + str(mode))
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
            return
        
        self.run_on(self.df)
        
        if self.error==1:
            return
        
        nrow = len(self.df_np)
        #self.df_np[self.col1] = self.df[self.col1]
        
        # register and save new columns one by one
        for i in range(0,self.numbers_count):
            fld = self.cols[i]
            fname = fld + '.csv'
            self.df_np[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

        self.os.remove(workdir + self.temp_file_name)
        
    def apply(self, df_add):
        self.run_on(df_add)
  
        for i in range(0,self.numbers_count):
            fld = self.cols[i]
            df_add[fld] = self.df_np[fld]
    
agent_{id} = cls_agent_{id}()
