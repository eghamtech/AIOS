#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv','dict_field1|dict_field1.csv','dict_field2|dict_field2.csv']
#key=tfhub_model;  type=constant;  value="https://tfhub.dev/google/universal-sentence-encoder-large/5"
#key=col_max_length;  type=constant;  value=200
#key=new_field_prefix;  type=constant;  value=tfhub_vecs
#key=field_prefix_use_source_names;  type=constant;  value=True
#key=include_columns_type;  type=constant;  value=is_dict_only
#key=include_columns_containing;  type=constant;  value=
#key=ignore_columns_containing;  type=constant;  value='%ev_field%' and '%onehe_%'
#key=random_seed_init;  type=random_int;  from=1;  to=10000000;  step=1
#end_of_parameters

# AICHOO OS Simple Agent
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns from given fields by concatenating text and vectorising words
# vectorisation performed by pre-trained TF Hub model
# all source fields expected to be dictionary fields
#
# number of new columns created will be equal to embedding vector size
#
# if "fields_source" parameter not specified then 2 fields will be obtained randomly
# according to normal AIOS logic

import warnings
warnings.filterwarnings("ignore")
import gc
gc.collect()

import pandas as pd
import numpy  as np
import tensorflow as tf
import tensorflow_hub as hub
import os.path, bz2, pickle, re

from datetime import datetime

class cls_agent_{id}:
    with tf.device('CPU'):
        tfhub_nlp = hub.load({tfhub_model})

    data_defs = {fields_source}

    # obtain a unique ID for the current instance
    result_id         = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix              = "{new_field_prefix}"
    field_prefix_use_source_names = {field_prefix_use_source_names}

    col_max_length    = {col_max_length}
    agent_name        = 'agent_' + str(result_id)
    rn_seed_init      = {random_seed_init}

    dicts_agent    = {}
    new_columns    = []
    dict_cols      = []
    dict_cols_full = []
    
 
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
        sfile = workdir + self.agent_name + '.model'
        if os.path.isfile(sfile):
            rfile = bz2.BZ2File(sfile, 'r')
            self.dicts_agent = pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.agent_name + ': TF Hub Vectors agent dictionaries model loaded')


    def run_on(self, df_run, apply_fun=False):
        self.new_columns = []

        # commented out because new data should already come with 'dict_' colums used by this agent
        # if apply_fun:
        #     for col_name in self.dicts_agent['dict_cols']:
        #         df_run['dict_'+col_name] = df_run[col_name]   # .map( self.dicts_agent[col_name] )   - new data should come as text, not dictionary key
        
        with tf.device('CPU'):
            doc = self.tfhub_nlp(['The quick brown fox jumps over the lazy dog.'])
        num_new_cols = len(doc[0])                     # establish length of TF model embeddings vector 

        for i in range(0,num_new_cols):
            new_col_name = self.new_field_prefix + '_vel_' + str(i) + '_' + str(self.result_id)
            self.new_columns.append(new_col_name)
        
        block_progress = 0
        total          = len(df_run)
        block          = int(total/50)
        df_new         = []

        for i, rowTuple in enumerate(df_run[self.dicts_agent['dict_cols_full']].itertuples(index=False)):
            row = ''
            for col in rowTuple:
                row += ' ' + str(col)
            row = row[1:]                   # remove trailing space after concatenations
        
            with tf.device('CPU'):
                doc = self.tfhub_nlp([row])   # vectorise row using TF Hub model
            row_v = list(doc.numpy()[0])      # convert tensor to numpy and put it to list

            df_new.append(row_v)

            block_progress += 1
            if (block_progress >= block):
                block_progress = 0
                print (str(datetime.now()), " rows processed: ", round((i+1)/total*100,0), "%")

        df_new = pd.DataFrame(df_new, columns=self.new_columns)

        if apply_fun:
            df_run[self.new_columns] = df_new
        else:
            # save and register each new column
            nrow = len(df_new)
            for i in range(0,len(self.new_columns)):
                fld   = self.new_columns[i]
                fname = fld + '.csv'
                df_new[[fld]].to_csv(workdir+fname)
                print ("#add_field:"+fld+",N,"+fname+","+str(nrow))
                    

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
                self.df['dict_'+col_name]  = self.df[col_name].map(dict_temp)

                self.dict_cols.append(col_name)
                self.dict_cols_full.append('dict_'+col_name)

        self.dicts_agent['dict_cols']      = self.dict_cols
        self.dicts_agent['dict_cols_full'] = self.dict_cols_full

        self.run_on(self.df)
        nrow = len(self.df)

        self.dicts_agent['new_columns'] = self.new_columns
        # save dictionary of all auxiliary data into file
        sfile = bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        pickle.dump(self.dicts_agent, sfile)
        sfile.close()


    def apply(self, df_add):
        self.run_on(df_add, apply_fun=True)

agent_{id} = cls_agent_{id}()
