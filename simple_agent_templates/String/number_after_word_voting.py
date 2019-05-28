#start_of_parameters
#key=fields_source;  type=constant;  value='string_field|string_field.csv'
#key=words;  type=constant;  value=['No','Maybe','Yes']
#key=new_field_prefix;  type=constant;  value=vote_
#key=include_columns_type;  type=constant;  set='is_dict_only'
#key=include_columns_containing;  type=constant;  set=
#key=ignore_columns_containing;  type=constant;  set='%ev_field%' and '%onehe_%' and '%scaled%'
#end_of_parameters

# AICHOO OS Simple Agent
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new column from given string field by finding numbers after specific words
# and applying voting function to such numbers
#
# if "fields_source" is empty then source field is assigned by AIOS - uncomment line 27 and comment 28 in such case

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import os.path, bz2, pickle, re

    #col_definition1 = "{random_field_distinct}"         # use this line for random field selection
    col_definition1 = "{fields_source}"                  # fixed field selection
    col1  = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    
    words = {words}
    dicts_agent = {}
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
    output_column     = new_field_prefix + col1 + '_' + str(result_id)
    agent_name        = 'agent_' + str(result_id)
    
    def __init__(self):
        from datetime import datetime
        
        if self.os.path.isfile(workdir + self.output_column + '_dicts.model'):
            rfile = self.bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'r')
            self.dicts_agent = self.pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.output_column + ' dictionaries model loaded')
            
    def is_set(self, s):
        return len(s)>0 and s!="0"
    
    def tokenize(self, s):
        swords = ''
        if type(s)==str:
           swords = ' '.join(self.re.findall(r"[\w'`]+|[.,!?;]", s))

        return swords 
        
    def get_number_after_word(self, word, sentence):
        s_tok = self.tokenize(sentence)                   # split string into words with punctuation
        match = self.re.search(word + ' (\d+)', s_tok)    # search for number after a word

        if match == None:
            match = self.re.search(word, s_tok)           # just match a word when no number after the word is found

            if match == None:
                ret = 0                                   # if no word found then return 0
            else:
                ret = 1                                   # if word is found without a number after it return 1
        else:
            ret = int(match.group(1))                     # return actual number found after the word

        return ret

    def get_vote(self, columns, row):
        # weights based score calculation
        wts = -1*row[columns[0]] + 0*row[columns[1]] + 1*row[columns[2]]
        n   = row[columns[0]] + row[columns[1]] + row[columns[2]]
        fr  = wts/n
        
        ret = 0      
        if n != 0:
            fr  = wts/n
            if n <= 2:
                if fr <= -0.5:
                    ret = 1
                elif fr > -0.5 and fr <= 0.5:
                    ret = 2
                elif fr > 0.5:
                    ret = 3
            else:
                if fr <= -0.5:
                    ret = 1
                elif fr > -0.5 and fr < 0.5:
                    ret = 2
                elif fr >= 0.5:
                    ret = 3
        
        return ret
                

    def run_on(self, df_run):
        col_name = self.col1
        
        df_t = df_run[[col_name]]
        df_t['dict_' + col_name] = df_t[col_name].map(self.dicts_agent[col_name])
        
        for word in self.words:
            # create column for each word with a number found for such word
            df_t[word] = df_t['dict_' + col_name].apply( (lambda x: self.get_number_after_word(word, x)) )
            
        # apply voting function to each row of given words' numbers
        df_run[self.output_column] = df_t[self.words].apply( (lambda x: self.get_vote(self.words, x)), axis=1 )
 

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df  = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        unique_l = self.df[self.col1].unique()

        if len(unique_l) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")
            return

        file_name  = self.file1
        col_name   = self.col1

        if self.os.path.isfile(workdir + 'dict_' + file_name):
            # load dictionary if it exists
            self.dicts_agent[col_name] = self.pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
  
        self.run_on(self.df)
        nrow = len(self.df)

        # save new column into CSV file
        fld   = self.output_column
        fname = fld + '.csv'
        self.df[[fld]].to_csv(workdir+fname)
        
        # save dictionaries of the new column into model file needed for Apply stage
        sfile = self.bz2.BZ2File(workdir + fld + '_dicts.model', 'w')
        self.pickle.dump(self.dicts_agent, sfile) 
        sfile.close()
            
        print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add)


agent_{id} = cls_agent_{id}()
