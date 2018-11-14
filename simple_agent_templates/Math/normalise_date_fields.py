#start_of_parameters
#key=fields_source;  type=constant;  value=['subtrahend_field|subtrahend_field.csv','minuend_field1|minuend_field1.csv','minuend_field2|minuend_field2.csv']
#key=map_name;  type=constant;  value=Y_map
#key=new_field_prefix; type=constant;  value='map_year_'
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent maps the first field in "field_source" to a value from "map_name" for each row
# and subtracts such value from all other columns in "field_source" 
# if "map_name" is "dummy_map" then the value to substract is as given in the first field of "field_source"

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    dummy_map = {}
            
    Y_map = {
        520115:2017,
        520116:2018,
        520117:2019,
        520118:2020,
        520119:2021
    }
    
    TS_map = {
        520115:1483228800,
        520116:1514764800,
        520117:1546300800,
        520118:1577836800,
        520119:1609459200
    }

    import pandas as pd
    import numpy as np
    
    data_defs = {fields_source}
    data_map = {map_name}
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix = {new_field_prefix} + str(result_id) + '_'

    def run_on(self, df_run):      
        subtrahend_col_name = self.data_defs[0].split("|")[0]
        subtrahend_col_name_mapped = subtrahend_col_name + '_mapped'
        if data_map == dummy_map:
            subtrahend_col_name_mapped = subtrahend_col_name
        else:
            df_run[subtrahend_col_name_mapped] = df_run[subtrahend_col_name].map(self.data_map)
        
        for i in range(1,len(self.data_defs)):
            minuend_col = self.data_defs[i].split("|")[0]
            col_res = self.new_field_prefix + minuend_col 
            df_run[col_res] = df_run[minuend_col] - df_run[subtrahend_col_name_mapped]
              
        
    def run(self, mode):
        print ("enter run mode " + str(mode))
        
        for i in range(0,len(self.data_defs)):
             col_name = self.data_defs[i].split("|")[0]
             file_name = self.data_defs[i].split("|")[1]
    
             if i==0:
                self.df = self.pd.read_csv(workdir+file_name)[[col_name]]
             else:
                self.df = self.df.merge(self.pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)
        
        nrow = len(self.df)
 
        self.run_on(self.df)
    
        is_dict="N"
        for i in range(1,len(self.data_defs)):
            minuend_col = self.data_defs[i].split("|")[0]
            col_res = self.new_field_prefix + minuend_col 
            col_res_file = col_res+'.csv'
            self.df[[col_res]].to_csv(workdir+col_res_file)
            print ("#add_field:"+col_res+","+is_dict+","+col_res_file+","+str(nrow)+",Y")

        
    def apply(self, df_add):
        self.run_on(df_add)
 
agent_{id} = cls_agent_{id}()
