#start_of_parameters
#key=field_source;  type=constant;  value=enter_source_field_name_with_csv_file_name
#key=field_filter;  type=constant;  value=enter_field_to_filter_by_with_csv_file_name
#key=filter_values;  type=constant;  value=enter_values_to_filter_by
#key=set_value;  type=constant;  value=enter_value_to_set
#key=new_field_prefix;  type=constant;  value=filter_
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new column which is a copy of "field_source" but with some rows set to "set_value" 
# where those rows in "field_filter" appear in "filter_values" 
# if only "field_source" specified then create new column as exact its copy

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import numpy as np
    import os.path
    
    # data_defs          = ["{field_source}","{field_filter}"]
    filter_values_list = [{filter_values}]
    new_value          = {set_value}
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix = "{new_field_prefix}"
   
    def is_set(self, s):
        return len(s)>0 and s!="0"
    
    def __init__(self):
        self.data_defs = []
        
        if self.is_set("{field_source}") and self.is_set("{field_filter}"):
            self.data_defs = ["{field_source}","{field_filter}"]
        elif self.is_set("{field_source}"):
            self.data_defs = ["{field_source}"]
            
        self.output_column    = self.new_field_prefix + self.data_defs[0].split("|")[0] + "_" + str(self.result_id)
        self.output_filename  = self.output_column + ".csv"

                   
    def run(self, mode):
        print ("enter run mode " + str(mode))
        cols_dicts = {}
        
        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if i==0:
                self.df = self.pd.read_csv(workdir+file_name)[[col_name]]
            else:
                self.df = self.df.merge(self.pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)
                
            # load dictionary if it exists
            if self.os.path.isfile(workdir + 'dict_' + file_name):
                dict_temp = self.pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
                cols_dicts[col_name]      = dict_temp
                self.df["dict_"+col_name] = self.df[col_name].map(dict_temp)
                
        nrow = len(self.df)
        
        if len(self.data_defs) == 1:
            # only source field specified - just copy source field into target
            col_source = self.data_defs[0].split("|")[0]
            self.df[self.output_column] = self.df[col_source]
            
            is_dict="N"
            if cols_dicts.get(col_source) != None:
                is_dict="Y"
                output_dict = cols_dicts.get(col_source)
                self.pd.DataFrame(list(output_dict.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+self.output_filename, encoding='utf-8')
                
            self.df[[self.output_column]].to_csv(workdir+self.output_filename)
            print ("#add_field:"+self.output_column+","+is_dict+","+self.output_filename+","+str(nrow)+",N")
        else:
            # fiter field specified - copy and filter
            col_source = self.data_defs[0].split("|")[0]
            col_filter = self.data_defs[1].split("|")[0]

            self.df[self.output_column] = self.df[col_source]
            self.df.loc[self.np.isin(self.df[col_filter], self.filter_values_list), self.output_column] = self.new_value

            self.df[[self.output_column]].to_csv(workdir+self.output_filename)
            print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(nrow)+",N")

        
    def apply(self, df_add):
        if len(self.data_defs) == 1:
            col_source = self.data_defs[0].split("|")[0]
            df_add[self.output_column] = df_add[col_source]
        else:
            col_source = self.data_defs[0].split("|")[0]
            col_filter = self.data_defs[1].split("|")[0]

            df_add[self.output_column] = df_add[col_source]
            df_add.loc[self.np.isin(df_add[col_filter], self.filter_values_list), self.output_column] = self.new_value
       

agent_{id} = cls_agent_{id}()
