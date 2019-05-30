#start_of_parameters
#key=fields_source;  type=constant;  value=enter_source_field_name_with_csv_file_name
#key=fields_filter;  type=constant;  value=enter_fields_to_filter_by_with_csv_file_name
#key=filter_condition;  type=constant;  value=enter_condition_to_filter_by
#key=set_value;  type=constant;  value=enter_value_to_set
#key=set_value_default;  type=constant;  value=
#key=new_field_prefix;  type=constant;  value=filter_
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new column which is a copy of "field_source" but with some rows set to "set_value" 
# where those rows in "fields_filter" match condition as specified in "filter_condition" with reference to df_run dataframe 
# 
# if "set_value_default" is set then new column is set to it for inversed "filter_condition"

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import numpy as np
    import os.path
    
    col_definition1 = "{fields_source}"                  # fixed field selection
    col1  = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    
    data_defs          = [{fields_filter}]
    new_value          = {set_value}
    default_value      = {set_value_default}
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix = "{new_field_prefix}"
    output_column    = new_field_prefix + col1 + '_' + str(result_id)
    output_filename  = output_column + ".csv"
    agent_name       = 'agent_' + str(result_id)
   
    def is_set(self, s):
        return len(s)>0 and s!="0"
    

    def run_on(self, df_run):
        col_source = self.col1
                
        if self.is_set(self.default_value):
            df_run[self.output_column] = self.default_value
        else:
            df_run[self.output_column] = df_run[col_source]
                      
        df_run.loc[{filter_condition}, self.output_column] = self.new_value

                   
    def run(self, mode):
        print ("enter run mode " + str(mode))
        
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            self.df = self.df.merge(self.pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)
            
            # load dictionary if it exists
            if self.os.path.isfile(workdir + 'dict_' + file_name):
                dict_temp = self.pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
                self.df["dict_"+col_name] = self.df[col_name].map(dict_temp)
                
        self.run_on(self.df)
        nrow = len(self.df)

        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(nrow)+",N")

        
    def apply(self, df_add):
        self.run_on(df_add)
       

agent_{id} = cls_agent_{id}()
