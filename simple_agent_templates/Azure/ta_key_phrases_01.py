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
# this agent creates new columns by hot-encoding key phrases obtained from Azure Text Analytics API

if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import re, bz2, pickle, os.path

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'azure_ta_key_phrases_' + col1 + '_'
    temp_file_name = field_prefix + '.tmp'
    fldprefix = field_prefix + str(result_id)
    
    max_phrases = 20
    azure_subscription_key = "fcbbbd8a5afb4bf7b8bc24013bf1f2de"
    text_analytics_base_url = "https://uksouth.api.cognitive.microsoft.com/text/analytics/v2.0/"
    azure_headers = {"Ocp-Apim-Subscription-Key": azure_subscription_key}    
    
    error = 0
    
    def printlog(self, mesg):
        from datetime import datetime
        global DEBUG
        if DEBUG == 1:
            print (str(datetime.now()), mesg)
            
    def __init__(self):
        global dicts         
        self.cols = []
        
        # load dictionary of text records
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        
        # if saved model for phrases already exists then load it from filesystem
        if self.os.path.isfile(workdir + self.fldprefix + '_phrases.model'):
            rfile = self.bz2.BZ2File(workdir + self.fldprefix + '_phrases.model', 'r')
            self.all_phrases_dict = self.pickle.load(rfile)
            rfile.close()
        
        # load dictionary of language names
        if self.os.path.isfile(workdir + 'dict_' + self.fldprefix + '_lang.csv'):
            self.dict_lang = self.pd.read_csv(workdir + 'dict_' + self.fldprefix + '_lang.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
            
        
    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
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
    
    def _requests_post(self, url, headers, json, index):
        import requests
        attempts = 0
        not_successful = True
        r = requests.Response()
        
        while not_successful and attempts < 10:
           try:
              r = requests.post(url, headers=headers, json=json)
              r.raise_for_status()
              not_successful = False
           except requests.exceptions.RequestException as e:
              attempts += 1
              print (e)
              if attempts < 10:
                 print ('Error Azure request at row: ', index, '; retry attempt: ', attempts)
              else:
                 print ('Error Azure request at row: ', index, '; FATAL no more attempts')
                 self.error = 1
                 return
        return r
    
    def run_on(self, df_run, mode):
        import json
        if self.index_from == 0:
            # create dataframe with actual texts column using saved dictionary
            self.dfx = self.pd.DataFrame()
            self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
    
        azure_block = 10               # defines chunks of texts to be sent to Azure at once 
        i = 0
       
        azure_docs = []
        for index, row in self.dfx.iloc[self.index_from:self.index_to].iterrows():
            i+=1                       # need this counter to group records into chunks to be sent to Azure API
            if i == 1:                 # get chunk starting index at the beginning of the chunk
                lblock_start = index   
            
            sline1 = row[self.col1]
            # create a list of documents as required by Azure e.g.
            # { 'id': '0', 'text': 'Este es un document escrito en Español.' }
            azure_docs.append( { 'id': str(index), 'text': sline1 } )
            
            if i >= azure_block or index == (self.index_to-1):
                # first part is to identify languages for each record
                r = self._requests_post(self.text_analytics_base_url + "languages", headers=self.azure_headers, json={'documents':azure_docs}, index=index)
                self.langs = r.json()
                #pprint(self.langs)
                
                for document in self.langs['documents']:
                    _id = int(document['id'])
                    _lang = document["detectedLanguages"][0]['iso6391Name']
                    _score = document["detectedLanguages"][0]['score']
                    
                    self.dfx.at[_id, self.fldprefix + '_lang'] = _lang
                    self.dfx.at[_id, self.fldprefix + '_lang_score'] = _score
                
                azure_docs = []
                if index == (self.index_to-1):
                    lblock_to = self.index_to
                else:
                    lblock_to = lblock_start+azure_block
                
                # second part create list of documents with languages e.g.,
                # { 'id': '0', 'language': 'es', 'text': 'Este es un document escrito en Español.' }
                for index, row in self.dfx.iloc[lblock_start:lblock_to].iterrows():
                    sline1 = row[self.col1]
                    azure_docs.append( {'id': str(index), 'language': row[self.fldprefix + '_lang'], 'text': sline1} )
                #pprint(azure_docs)
                    
                r = self._requests_post(self.text_analytics_base_url + "keyPhrases", headers=self.azure_headers, json={'documents':azure_docs}, index=index)
                self.phrases = r.json()
                #pprint(self.phrases)
                
                for document in self.phrases['documents']:
                    self.all_phrases.append(document)
                
                if mode == 0:
                    # invoked during training data conversion
                    # save accumulated list of documents with phrases in case agent crashes
                    # to avoid calling Azure API on already processed records after restart
                    dict_temp = {}
                    dict_temp['all_phrases'] = self.all_phrases
                    dict_temp['dfx'] = self.dfx
                    sfile = self.bz2.BZ2File(workdir + self.temp_file_name, 'w')
                    self.pickle.dump(dict_temp, sfile) 
                    sfile.close()          
                    print ('phrases saved to temp file; length: ', len(self.all_phrases))
                
                i = 0                  # reset counter to start new chunk 
        
        if mode == 0:
            # invoked during training data conversion
            # extract a list of all phrases discovered by Azure API
            all_phrases_list = []
            for document in self.all_phrases:
                all_phrases_list.extend(document['keyPhrases'])

            # create a list of unique phrases sorted in descending order of frequency 
            # and limit to max_phrases most frequent ones
            all_phrases_list = self.pd.DataFrame(all_phrases_list, columns=['phrases'])
            all_phrases_list = all_phrases_list['phrases'].value_counts().index.tolist()[0:self.max_phrases]
            # create an index of all unique phrases
            keys1 = range(0, len(all_phrases_list))
            self.all_phrases_dict = dict(zip(all_phrases_list, keys1))
            
            # save dictionary to file so it can be used during applying to new data
            sfile = self.bz2.BZ2File(workdir + self.fldprefix + '_phrases.model', 'w')
            self.pickle.dump(self.all_phrases_dict, sfile) 
            sfile.close()          
        
        # prepare list of new columns and add columns to temp dataframe
        self.cols.append(self.fldprefix + '_lang')
        self.cols.append(self.fldprefix + '_lang_score')
        for i in range(0,self.max_phrases):
            fld = self.fldprefix + '_' + str(i)
            self.cols.append(fld)
            self.dfx[fld] = 0
        
        # iterate through all records and hot encode each phrase to columns
        for document in self.all_phrases:
            _id = int(document['id'])
            _phrases = document['keyPhrases']
            for phr in _phrases:                             # iterate through each phrase in current record
                ind = self.all_phrases_dict.get(phr,-1)      # see if phrase is in dictionary
                if ind in range(0,self.max_phrases):         # set phrase ID's corresponding column to 1
                    self.dfx.at[_id, self.fldprefix + '_' + str(ind)] = 1
          
        # append all newly created columns to supplied dataframe
        for fld in self.cols:
            df_run[fld] = self.dfx[fld]
            
        # during applying to new data convert language name to its index
        # using dictionary saved during training
        if mode == 1:
            dict_lang_modified = False
            fld = self.fldprefix + '_lang'
            for index, row in df_run.iterrows(): 
                lang = row[fld]
                if not (lang in self.dict_lang):                              # if value in current row and column not in dictionary
                    self.printlog ("Azure Key Phrases: column " + fld + "; value: " + lang + " not in dictionary")
                    new_key = 1 + max(self.dict_lang.values())                # create new key with max+1 value
                    self.dict_lang[lang] = new_key                            # add text:key to original dictionary
                    df_run.at[index, fld] = new_key
                    dict_lang_modified = True
                else:
                    df_run.at[index, fld] = self.dict_lang[lang]
            
            if dict_lang_modified:
                self.pd.DataFrame(list(self.dict_lang.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+fld+'.csv', encoding='utf-8')
                
        
    def run(self, mode):
        print ("enter run mode " + str(mode))     
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
            return
          
        self.all_phrases = []
        self.index_from = 0
        self.index_to = 61  # = len(self.df)
        
        # if saved temp object exists then load it from filesystem to carry on from last good batch
        if self.os.path.isfile(workdir + self.temp_file_name):   
            rfile = self.bz2.BZ2File(workdir + self.temp_file_name, 'r')
            dtmp = self.pickle.load(rfile)
            rfile.close()
            
            self.all_phrases = dtmp['all_phrases']
            self.dfx = dtmp['dfx']
            
            self.index_from = len(self.all_phrases)
            print ('phrases array loaded from temp file, continue conversion from row: ', self.index_from+1)
                   
        self.run_on(self.df, mode=0)
        
        if self.error==1:
            return
        
        nrow = len(self.df)
        # register and save new columns one by one        
        for fld in self.cols:
            fname = fld + '.csv'
            if fld == (self.fldprefix + '_lang'):
                self.dict_lang = self.make_dict(self.df[fld].fillna(''))             # create dictionary of given text column  
                self.df[fld] = self.df[fld].fillna('').map(self.dict_lang)           # replace column values with corresponding values from dictionary
                # save dictionary for each text column into separate file
                self.pd.DataFrame(list(self.dict_lang.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+fld+'.csv', encoding='utf-8')
                # save column of indexes
                self.df[[fld]].to_csv(workdir+fname)
                print ("#add_field:"+fld+",Y,"+fname+","+str(nrow))
            else:
                self.df[[fld]].to_csv(workdir+fname)
                print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


        
    def apply(self, df_add):
        self.run_on(df_add, mode=1)
  
    
agent_{id} = cls_agent_{id}()
