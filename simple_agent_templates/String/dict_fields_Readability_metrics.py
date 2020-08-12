#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv','dict_field1|dict_field1.csv','dict_field2|dict_field2.csv']
#key=duplicate_text_times;  type=constant;  value=1
#key=col_max_length;  type=constant;  value=200
#key=new_field_prefix;  type=constant;  value=read_metrics_
#key=field_prefix_use_source_names;  type=constant;  value=True
#key=include_columns_type;  type=constant;  value=is_dict_only
#key=include_columns_containing;  type=constant;  value=
#key=ignore_columns_containing;  type=constant;  value='%ev_field%' and '%onehe_%'
#key=out_file_extension;  type=constant;  value=.csv.bz2
#end_of_parameters

# AICHOO OS Simple Agent
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns from given fields by concatenating text and 
# calculating readability metrics described in https://py-readability-metrics.readthedocs.io/en/latest/
#
# all source fields expected to be dictionary fields, but non-dictionary fields will be converted to text
# concatenated text could be repeated by specified number of times separated by space to deal with texts smaller than 100 words
# number of new columns created will be the same as number of metrics 
#
# if "fields_source" parameter not specified then 2 fields will be obtained randomly
# according to normal AIOS logic

import warnings
warnings.filterwarnings("ignore")
import gc
gc.collect()

import pandas as pd
import os.path, bz2, pickle, re

from datetime import datetime
from readability import Readability

class cls_agent_{id}:
    data_defs = {fields_source}

    # obtain a unique ID for the current instance
    result_id         = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
    col_max_length    = {col_max_length}
    agent_name        = 'agent_' + str(result_id)

    field_prefix_use_source_names = {field_prefix_use_source_names}
    duplicate_text_times          = {duplicate_text_times}
    out_file_extension            = "{out_file_extension}"

    #dicts_agent = {}
    new_columns = []
    dict_cols   = []
    
    # All Readability metrics to be calculated
    metrics = { 
        'flesch'                :'r.flesch().score',
        'flesch_kincaid'        :'r.flesch_kincaid().score',
        'gunning_fog'           :'r.gunning_fog().score',
        'coleman_liau'          :'r.coleman_liau().score',
        'dale_chall'            :'r.dale_chall().score',
        'ari'                   :'r.ari().score',
        'linsear_write'         :'r.linsear_write().score',
        'smog'                  :'r.smog().score',
        'spache'                :'r.spache().score',
        'num_letters'           :"r.statistics()['num_letters']",
        'num_words'             :"r.statistics()['num_words']",
        'num_sentences'         :"r.statistics()['num_sentences']",
        'num_polysyllabic_words':"r.statistics()['num_polysyllabic_words']",
        'avg_words_per_sentence':"r.statistics()['avg_words_per_sentence']",
        'avg_syllables_per_word':"r.statistics()['avg_syllables_per_word']"          
    }   

    def is_set(self, s):
        return len(s)>0 and s!="0"

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))

    def __init__(self):
        #if not self.is_set(self.data_defs):
        #    self.data_defs = ["{_random_dict_distinct}", "{_random_dict_distinct}"]

        if self.field_prefix_use_source_names:                   
            # concatenate all source column names into new field prefix
            col_max_length = int(200 / len(self.data_defs))             # allow 200 characters max combined col name length
            for i in range(0,len(self.data_defs)):
                col_name = self.data_defs[i].split("|")[0]
                col_name = col_name[:col_max_length]                   # only take first col_max_length chars from each column
                self.new_field_prefix = self.new_field_prefix + '_' + col_name

        # if saved dictionaries for the target field already exist then load them from filesystem
        # sfile = workdir + self.agent_name + '.model'
        # if os.path.isfile(sfile):
        #     rfile = bz2.BZ2File(sfile, 'r')
        #     self.dicts_agent = pickle.load(rfile)
        #     rfile.close()
        #     print (str(datetime.now()), self.agent_name + ': SpaCy NER Stats agent dictionaries model loaded')


    def run_on(self, df_run, apply_fun=False):
        # if apply_fun:
        #     for col_name in self.dicts_agent['dict_cols']:
        #         df_run['dict_'+col_name] = df_run[col_name]   # .map( self.dicts_agent[col_name] )   - new data should come as text, not dictionary key

        self.dict_cols = []
        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            self.dict_cols.append(col_name)

        self.new_columns = []
        for key in self.metrics:
            # all metric names should be stored in this dictionary, so just iterate over them
            new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_' + str(key)
            self.new_columns.append(new_col_name)

        block_progress = 0
        total = len(df_run)
        block = int(total/50)
        
        df_metrics = []
        for index, row in df_run.iterrows():
            row_str = ''
            for col_name in self.dict_cols:
                row_str += ' ' + str(row[col_name])   # concatenate columns into one string

            row_str = row_str * self.duplicate_text_times   # repeat text number of times - may be needed if data is typically less than 100 words
            row_str = row_str[1:]                           # remove space added during columns concatenation
            
            r = Readability(row_str)

            row_metrics = []
            for key in self.metrics:
                try:
                    mscore = eval(self.metrics[key])
                except Exception as e:
                    mscore = float('nan')

                row_metrics.append(mscore)

            df_metrics.append(row_metrics)  

            block_progress += 1
            if (block_progress >= block):
                block_progress = 0
                print (str(datetime.now()), " rows processed: ", round((index+1)/total*100,0), "%")

        df_run[self.new_columns] = pd.DataFrame(df_metrics)   


    def run(self, mode):
        print ("enter run mode " + str(mode))

        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if i==0:
                self.df = pd.read_csv(workdir+file_name)[[col_name]]
            else:
                self.df = self.df.merge(pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)

            if os.path.isfile(workdir + 'dict_' + file_name):
                # load dictionary if it exists
                dict_temp = pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()

                #self.dicts_agent[col_name] = dict_temp
                #self.df['dict_'+col_name]  = self.df[col_name].map(dict_temp)
                self.df[col_name]  = self.df[col_name].map(dict_temp)

                #self.dict_cols.append(col_name)

        #self.dicts_agent['dict_cols'] = self.dict_cols

        self.run_on(self.df)
        nrow = len(self.df)

        # self.dicts_agent['new_columns'] = self.new_columns
        # # save dictionary of all auxiliary data into file
        # sfile = bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        # pickle.dump(self.dicts_agent, sfile)
        # sfile.close()

        # save and register each new column
        for i in range(0,len(self.new_columns)):
            fld   = self.new_columns[i]
            fname = fld + self.out_file_extension
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add, apply_fun=True)

agent_{id} = cls_agent_{id}()
