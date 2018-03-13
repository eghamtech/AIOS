#start_of_parameters
#key=field_source;  type=constant;  value=enter_source_field_name_with_csv_file_name
#key=field_filter;  type=constant;  value=enter_field_to_filter_by_with_csv_file_name
#key=filter_values;  type=constant;  value=enter_values_to_filter_by
#key=set_value;  type=constant;  value=enter_value_to_set
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent 

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import numpy as np
    
    data_defs = ["{field_source}","{field_filter}"]
    filter_values_list = [{filter_values}]
    new_value = {set_value}
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix = "filter_"
    output_column = new_field_prefix + str(result_id)
    output_filename = output_column + ".csv"
    
    def __init__(self):
        for i in range(0,len(self.data_defs)):
             col_name = self.data_defs[i].split("|")[0]
             file_name = self.data_defs[i].split("|")[1]
    
             if i==0:
                self.df = self.pd.read_csv(workdir+file_name)[[col_name]]
             else:
                self.df = self.df.merge(self.pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)
        
        
    def run(self, mode):
        print ("enter run mode " + str(mode))
        nrow = len(self.df)
        
        col_source = self.data_defs[0].split("|")[0]
        col_filter = self.data_defs[1].split("|")[0]
        
        self.df[self.output_column] = self.df[col_source]
        self.df.loc[self.np.isin(self.df[col_filter], self.filter_values_list), self.output_column] = self.new_value
       
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(nrow)+",Y")

        
    def apply(self, df_add):
        col_source = self.data_defs[0].split("|")[0]
        col_filter = self.data_defs[1].split("|")[0]
        
        df_add[self.output_column] = df_add[col_source]
        df_add.loc[self.np.isin(df_add[col_filter], self.filter_values_list), self.output_column] = self.new_value
       

agent_{id} = cls_agent_{id}()
