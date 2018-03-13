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
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # re-register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
        else:
            # re-register the same field as the source field exactly as it was, so nothing really changes
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",Y")

        
    def apply(self, df_add):
       # this method has nothing to do as this agent doesn't create its own fields
       return

agent_{id} = cls_agent_{id}()
