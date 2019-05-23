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
# if "fields_source" is empty then source field is assigned by AIOS

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
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
    new_field_name    = new_field_prefix + col1 + '_' + str(result_id)
    agent_name        = 'agent_' + str(result_id)
    
    def is_set(self, s):
        return len(s)>0 and s!="0"
    
    def tokenize(self, s):
        swords = ''
        if type(s)==str:
           swords = ' '.join(self.re.findall(r"[\w'`]+|[.,!?;]", s))

        return swords 
        
    def get_number_after_word(self, word, sentence):
        s_tok = self.tokenize(sentence)
        match = self.re.search(word + ' (\d+)', s_tok)

        if match == None:
            match = self.re.search(word, s_tok)

            if match == None:
                ret = 0
            else:
                ret = 1
        else:
            ret = int(match.group(1))

        return ret

    def get_vote(self, columns, row):
        wts = -1*row[columns[0]] + 0*row[columns[1]] + 1*row[columns[2]]
        n   = row[columns[0]] + row[columns[1]] + row[columns[2]]

        fr  = wts/n
        
        if n <= 2:
            if r <= -0.5:
                ret = 0
            elif r > -0.5 and r <= 0.5:
                ret = 1
            elif r > 0.5:
                ret = 2
        else:
            if r <= -0.5:
                ret = 0
            elif r > -0.5 and r < 0.5:
                ret = 1
            elif r >= 0.5:
                ret = 2
        
        return ret
                

    def run_on(self, df_run):
        col_name = self.col1
        
        df_t = df_run[[col_name]]
        
        for word in self.words:
            df_t[word] = df_t[col_name].apply( (lambda x: self.get_number_after_word(word, x)) )
            
        df_run[self.new_field_name] = df_t[self.words].apply( (lambda x: self.get_vote(self.words, x)) )
 

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df  = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        unique_list = self.df[self.col1].unique()

        if len(unique_list) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")
            return

        file_name  = self.file1
        col_name   = self.col1

        if self.os.path.isfile(workdir + 'dict_' + file_name):
            # load dictionary if it exists
            dict_temp         = self.pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
            self.df[col_name] = self.df[col_name].map(dict_temp)
  
        self.run_on(self.df)
        nrow = len(self.df)

        fld   = self.new_field_name
        fname = fld + '.csv'
        self.df[[fld]].to_csv(workdir+fname)
        print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add)


agent_{id} = cls_agent_{id}()
