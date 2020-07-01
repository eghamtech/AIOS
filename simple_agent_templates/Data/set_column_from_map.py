#start_of_parameters
#key=fields_source;  type=constant;  value=['f0|f0.csv','f1|f1.csv','f2|f2.csv']
#key=fields_source_group_map;  type=constant;  value=[(f0_value1,1),(f0_value2,2)]
#key=groups_map_nan;  type=constant;  value=enter_groups_map_nan
#key=nan_col;  type=constant;  value=enter_nan_col
#key=map_name;  type=constant;  value=class07_map
#key=field_prefix;  type=constant;  value=map_class07_
#key=output_str_type;  type=constant;  value=False
#key=out_file_extension;  type=constant;  value=.csv.bz2
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new column which is a derivative of columns "field_source" but with rows 
# set to value from the "map_name" depending on a value read from one of the columns from "fields_source"
# which itself is selected depending on value in the first column of "fields_source"
#
# "groups_map_nan" determines which groups from "field_source" to map NaN values for separately from "map_name"
# and "nan_col" determines which column to use to replace NaN values with
#
# "output_str_type" defines whether new column is dictionary (text) or numeric

import warnings
warnings.filterwarnings("ignore")
import os.path, bz2, pickle
import pandas as pd
import numpy as np
from datetime import datetime

class cls_agent_{id}:    
    dummy_map   = {}
    dicts_agent = {}
        
    class07_map = {    
        # maps [1,2]->0, [3]->1, [4]->2, [5,6,7]->3
        0:float('nan'),
        1:0,
        2:0,
        3:1,
        4:2,
        5:3,
        6:3,
        7:3
    }
    
    data_defs          = {fields_source}
    data_defs_groups   = dict( {fields_source_group_map} )
    output_str_type    = {output_str_type}
    out_file_extension = "{out_file_extension}"

    data_map       = {map_name}
    groups_map_nan = {groups_map_nan}
    nan_col        = {nan_col}

    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix = "{field_prefix}"
    # output_column = new_field_prefix + str(result_id)
    
    output_column    = new_field_prefix + data_defs[0].split("|")[0] + "_" + str(result_id)
    output_filename  = output_column + out_file_extension
   
    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def __init__(self):
        model_file = workdir + self.output_column + '_dicts.model'         
        if os.path.isfile(model_file):
            rfile = bz2.BZ2File(model_file, 'r')
            self.dicts_agent = pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.output_column + ' dictionaries model loaded')

    def run_on(self, df_run, mode_apply=False):
        columns = []
        for i in range(0,len(self.data_defs)):
             columns.append( self.data_defs[i].split("|")[0] )
        
        nrow = len(df_run)     
        output_list = []
        
        for index, row in df_run.iterrows():
            if len(self.data_defs) == 1:         # if only one column given then just map this column
                source_v = row[columns[0]]                                  # read actual value from given field
                output_v = self.data_map.get(source_v, float('nan'))        # map value to final result
                output_list.append(output_v)                
            else:                                # if more than one column given, proceed with field selection logic
                group = row[columns[0]]                                     # read value from the first field which determines which other field to read value from next
                group_col = self.data_defs_groups.get(group, 0)             # map first field's value to another field
                
                if group_col > 0 and group_col < len(columns):
                    source_v  = row[columns[group_col]]                         # read actual value from determined field
                    
                    if np.isnan(source_v) and (group in self.groups_map_nan):  # map NaN values to values in special column for specific groups
                        source_v = row[columns[self.nan_col]]
                        
                    output_v  = self.data_map.get(source_v, float('nan'))       # map value to final result
                else:
                    output_v  = float('nan')
                    
                output_list.append(output_v)
        
        df_run[self.output_column] = output_list
        
        if mode_apply:
            if self.output_str_type:
                df_run[self.output_column] = df_run[self.output_column].map(self.dicts_agent[self.output_column])
                
    def run(self, mode):
        print ("enter run mode " + str(mode))
        
        for i in range(0,len(self.data_defs)):
             col_name  = self.data_defs[i].split("|")[0]
             file_name = self.data_defs[i].split("|")[1]
    
             if i==0:
                self.df = pd.read_csv(workdir+file_name)[[col_name]]
             else:
                self.df = df.merge(pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)
        
        nrow = len(self.df)
 
        self.run_on(self.df)
    
        is_dict="N"
        if self.output_str_type:
             is_dict="Y"
             output_dict = self.make_dict(self.df[self.output_column])
             self.df[self.output_column] = self.df[self.output_column].map(output_dict)

             out_file = workdir + 'dict_' + self.output_filename
             pd.DataFrame(list(output_dict.items()), columns=['value', 'key'])[['key','value']].to_csv(out_file, encoding='utf-8')
            
             self.dicts_agent[self.output_column] = output_dict
            
             sfile = bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'w')
             pickle.dump(self.dicts_agent, sfile) 
             sfile.close()
                
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        print ("#add_field:"+self.output_column+","+is_dict+","+self.output_filename+","+str(nrow)+",N")

        
    def apply(self, df_add):
        self.run_on(df_add, mode_apply=True)

agent_{id} = cls_agent_{id}()
