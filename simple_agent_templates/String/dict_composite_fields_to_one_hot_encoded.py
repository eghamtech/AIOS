#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv','dict_field1|dict_field1.csv','dict_field2|dict_field2.csv']
#key=max_unique_values;  type=constant;  value=50
#key=col_max_length;  type=constant;  value=200
#key=clean_text_v;  type=constant;  value=0
#key=new_field_prefix;  type=constant;  value=onehe_mult_flds_
#key=include_columns_type;  type=constant;  set=is_dict_only
#key=include_columns_containing;  type=constant;  set=
#key=ignore_columns_containing;  type=constant;  set='%ev_field%' and '%onehe_%'
#end_of_parameters

# AICHOO OS Simple Agent
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns from given fields by hot encoding every unique value as 0 or 1
# all source fields expected to be dictionary fields and values may consist if multiple values 
# seperated by comma and other symbols
# 
# depending on value of clean_text_v composite values will be split only by comma or by more symbols and 
# treated as multi-valued records
#
# number of new columns created will be the same as number of unique values across all source fields
# if it is no larger than "max_unique_values"
# each column name will be suffixed with a corresponding string value
#
# if number of unique values exceed "max_unique_values" new columns will correspond to top "max_unique_values" most occurring
#
# if "fields_source" parameter not specified then 2 fields will be obtained randomly
# according to normal AIOS logic
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os.path, bz2, pickle, re
from datetime import datetime

    
class cls_agent_{id}:

    data_defs = {fields_source}

    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
    max_unique_values = {max_unique_values}
    col_max_length    = {col_max_length}
    clean_text_v      = {clean_text_v}
    agent_name        = 'agent_' + str(result_id)

    dicts_agent = {}
    new_columns = []
    dict_cols   = []

    def is_set(self, s):
        return len(s)>0 and s!="0"

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def clean_tokenize_text(self, str):
        if self.clean_text_v == 0:
            str = [x.strip() for x in re.split(',|,\s', str)]    # split by comma
        elif self.clean_text_v == 1:
            str = ''.join([i for i in str if not i.isdigit()])                    # strip numbers
            str = [x.strip() for x in re.split(';|,|-|\s|,\s|-\s|\s-\s', str)]    # split by comma and other chars and also strip whitespaces        
        return str

    def __init__(self):
        #if not self.is_set(self.data_defs):
        #    self.data_defs = ["{_andom_dict_distinct}", "{_andom_dict_distinct}"]       # adjust this line if random fields to be obtained

        # if saved dictionaries for the target field already exist then load them from filesystem
        sfile = workdir + self.agent_name + '.model'
        if os.path.isfile(sfile):
            rfile = bz2.BZ2File(sfile, 'r')
            self.dicts_agent = pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.agent_name + ': one hot encoding agent dictionaries model loaded')


    def run_on(self, df_run, apply_fun=False):
        self.new_columns = []

        if apply_fun:
            # when applying on new data, it comes with numerical IDs only, so map IDs to string values first
            for col_name in self.dicts_agent['dict_cols']:
                df_run['dict_'+col_name] = df_run[col_name].map( self.dicts_agent[col_name] )

        for k,v in self.dicts_agent['all_dicts'].items():
            # all allowed values should be stored in this dictionary as keys, so just iterate over them
            new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_' + re.sub('[^0-9a-zA-Z]+', '_', str(k))
            new_col_name = new_col_name[:self.col_max_length]
            self.new_columns.append(new_col_name)
            df_run[new_col_name] = 0

        for index, row in df_run.iterrows():
            # go through each column individually as specified previously
            for col_name in self.dicts_agent['dict_cols']:

                values_list = str(row['dict_'+col_name])                   # get string value which may consist of multiple values separated by comma
                values_list = self.clean_tokenize_text(values_list)

                for value in values_list:
                    if self.dicts_agent['all_dicts'].get(value) != None:
                        # if value found in dictionary, set corresponding column to 1
                        new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_' + re.sub('[^0-9a-zA-Z]+', '_', str(value))
                        new_col_name = new_col_name[:self.col_max_length]
                        df_run.at[index, new_col_name] = 1


    def run(self, mode):
        print ("enter run mode " + str(mode))
        all_values = []
        
        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if i==0:
                self.df = pd.read_csv(workdir+file_name)[[col_name]]
            else:
                self.df = self.df.merge(pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)

            sfile = workdir + 'dict_' + file_name     
            if os.path.isfile(sfile):
                # load dictionary if it exists
                dict_temp = pd.read_csv(sfile, dtype={'value': object}).set_index('key')["value"].to_dict()

                self.dicts_agent[col_name] = dict_temp            # save maps of each column for future use 
                self.df['dict_'+col_name]  = self.df[col_name].map(dict_temp)
                              
                for rowTuple in self.df[['dict_'+col_name]].itertuples(index=False):
                    row = str(rowTuple[0])                        # df should be just one column dataframe
                    row = self.clean_tokenize_text(row)
                    all_values.extend(row)                        # build up a list of all possible string values
                       
                self.dict_cols.append(col_name)
          
        
        self.df_all = pd.DataFrame(all_values, columns=['all_dict_columns'])
        unique_list = self.df_all['all_dict_columns'].unique()

        if len(unique_list) == 1:
            print ("Selected columns contains only 1 unique value - no point to do anything with it.")
            return

        # count unique values, sort in descending order and take only top max_unique_values
        df_top_values = self.df_all.groupby('all_dict_columns')['all_dict_columns'].agg({"count": len}).sort_values("count", ascending=False).head(self.max_unique_values).reset_index()
        # create new dictionary of top unique values
        dict_temp     = self.make_dict(df_top_values['all_dict_columns'].fillna(''))

        self.dicts_agent['dict_cols'] = self.dict_cols
        self.dicts_agent['all_dicts'] = dict_temp

        self.run_on(self.df)
        nrow = len(self.df)

        self.dicts_agent['new_columns'] = self.new_columns
        # save dictionary of all auxiliary data into file
        sfile = bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        pickle.dump(self.dicts_agent, sfile)
        sfile.close()

        # save and register each new column
        for i in range(0,len(self.new_columns)):
            fld   = self.new_columns[i]
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add, apply_fun=True)

agent_{id} = cls_agent_{id}()
