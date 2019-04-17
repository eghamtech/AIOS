#start_of_parameters
#key=max_unique_values;  type=constant;  value=50
#key=col_max_length;  type=constant;  value=200
#key=random_encoders_num;  type=constant;  value=5
#key=new_field_prefix;  type=constant;  value=one_random_
#key=include_columns_type;  type=constant;  set=
#key=include_columns_containing;  type=constant;  set=
#key=ignore_columns_containing;  type=constant;  set='%ev_field%' and '%one_random_%' and '%onehe%'
#end_of_parameters

# AICHOO OS Simple Agent
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns from given field by assigning randomly generated values to each unique value
#
# 'random_encoders_num' specifies number of new columns to be created
# only "max_unique_values" will be considered
#
# if number of unique values exceed "max_unique_values" such column will be binned into "max_unique_values" bins
# random values assigned to bins

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import os.path, bz2, pickle, re

    col_definition1 = "{random_field_distinct}"
    col1  = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]

    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
    max_unique_values = {max_unique_values}
    col_max_length    = {col_max_length}
    rnd_encoders_num  = {random_encoders_num}
    agent_name        = 'agent_' + str(result_id)

    dicts_agent = {}
    new_columns = []

    def is_set(self, s):
        return len(s)>0 and s!="0"

    def make_dict_random(self, dt, dt_num):
        dt_rand = {}
        for i in range(0,dt_num):
            # new dictionary of existing keys to random keys
            dt_rand[i] = {k:self.np.random_int("need to add range somehow") for k in dt.keys()}

        return dt_rand


    def __init__(self):
        from datetime import datetime
        # if saved dictionaries for the target field already exist then load them from filesystem
        if self.os.path.isfile(workdir + self.agent_name + '.model'):
            rfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'r')
            self.dicts_agent = self.pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.col1 + ': one hot encoding agent dictionaries model loaded')

    def run_on(self, df_run):
        col_name = self.col1
        self.new_columns = []

        if self.dicts_agent['dict_type'] == 'intervalindex':
            df_run['binned_'+col_name] = self.pd.cut(df_run[col_name], self.dicts_agent['intervals'])

        for k,v in self.dicts_agent[col_name].items():
            # this dictionary contains dictionary for every key transform, so just iterate over them
            new_col_name = self.new_field_prefix + str(self.result_id) + '_' + str(k) + '_' + col_name
            new_col_name = new_col_name[:self.col_max_length]
            self.new_columns.append(new_col_name)

            if self.dicts_agent['dict_type'] == 'dictionary':
                df_run[new_col_name] = df_run[col_name].map(v)


    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df  = self.pd.read_csv(workdir+self.file1)[[self.col1]]

        #self.df[self.col1] = self.df[self.col1].apply(lambda x: round(x,5) if self.pd.notnull(x) else None)
        self.df[self.col1] = int(self.df[self.col1] * 10000) / 10000
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
            dict_temp     = self.pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()

            df_top_values = self.df.groupby(col_name)[col_name].agg({"count": len}).sort_values("count", ascending=False).head(self.max_unique_values).reset_index()
            dict_temp     = {df_top_values[col_name][i]:dict_temp[df_top_values[col_name][i]] for i in range(0,len(df_top_values))}
            dict_rand     = self.make_dict_random(dict_temp, self.rnd_encoders_num)

            self.df["dict_"+col_name]     = self.df[col_name].map(dict_temp)
            self.dicts_agent['dict_type'] = 'dictionary'
            self.dicts_agent[col_name]    = dict_rand

        elif len(unique_list) <= self.max_unique_values:
            # create dictionary by iterating over unique values
            dict_temp = {x:str(x) for x in unique_list if str(x) != 'nan'}
            dict_rand = self.make_dict_random(dict_temp, self.rnd_encoders_num)

            self.df["dict_"+col_name]     = self.df[col_name].map(dict_temp)
            self.dicts_agent['dict_type'] = 'dictionary'
            self.dicts_agent[col_name]    = dict_rand

        else:
            # cut the column values into intervals
            df_cats   = self.pd.cut(self.df[col_name], self.max_unique_values)
            # convert intervals to strings and create a dictionary to make it compatible with dictionary approach
            dict_temp = df_cats.cat.categories.astype(str)
            dict_temp = {x:dict_temp[x] for x in range(0,len(dict_temp))}
            dict_rand = self.make_dict_random(dict_temp, self.rnd_encoders_num)

            self.dicts_agent['dict_type'] = 'intervalindex'
            self.dicts_agent[col_name]    = dict_rand
            self.dicts_agent['intervals'] = df_cats.cat.categories

        self.run_on(self.df)
        nrow = len(self.df)

        self.dicts_agent['new_columns'] = self.new_columns
        # save dictionary of all auxiliary data into file
        sfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        self.pickle.dump(self.dicts_agent, sfile)
        sfile.close()

        # save and register each new column
        for i in range(0,len(self.new_columns)):
            fld   = self.new_columns[i]
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)

            is_dict = "N"
            if self.dicts_agent['dict_type'] == 'dictionary':
                dict1 = { dict_rand[i][k]:v for k,v in dict_temp.items() }
                self.pd.DataFrame(list(dict1.items()), columns=['key', 'value']).to_csv(workdir+'dict_'+fname, encoding='utf-8')    #save new column dict
                is_dict = "Y"

            print ("#add_field:"+fld+","+is_dict+","+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add)


agent_{id} = cls_agent_{id}()
