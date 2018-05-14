#start_of_parameters
#key=source_filename;  type=constant;  value=enter_source_filename
#key=source_primary_field;  type=constant;  value=enter_source_primary_field
#key=primary_field;  type=constant;  value=enter_primary_field|its_filename
#key=field_prefix;  type=constant;  value=csv02
#key=target;  type=constant;  value=enter_target
#end_of_parameters

# This script will scan your CSV file for string columns, convert them to dictionaries
# and create columns in AIOS Memory with data for each column in the 'source_filename' CSV file.
# Provide correct 'source_filename' in the agent parameters.
# Variables 'workdir' and 'trainfile' (target filename) must be setup in 'Constants' area of AIOS
# Parameter 'target' specifies column to be marked as the prediction target.
#
# this version of the loader will scan existing data and will append new columns from "source_filename"
# where "source_primary_field" == "primary_field" column already in the AIOS Memory

if 'dicts' not in globals():
    dicts = {}  # dict of dicts. each of dicts has structure: key=string, value=number

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import re
    
    source_filename = "{source_filename}"
    source_primary_field = "{source_primary_field}"
    primary_field = "{primary_field}"
    new_field_prefix = "{field_prefix}"
    target = "{target}"
   
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    output_column = new_field_prefix + "_" + str(result_id)
    output_filename = output_column + ".csv"
    
    colmap = {}
        
    def __init__(self):
        global dicts
        self.df = self.pd.read_csv(workdir+self.source_filename, encoding='utf8')
        
        new_cols = []
        for c in self.df.columns:
              str1 = c
              str1 = self.re.sub('[^0-9a-zA-Z]+', '_', str1)
              str1 = output_column + "_" + str1
              new_cols.append(str1)
              self.colmap[c] = str1
        df.columns = new_cols
        
        self.char_cols = list(self.df.select_dtypes(include=['object']).columns)
        print ("source data loaded")
        print ("rows: ", len(self.df))
        print ("char columns:", self.char_cols)
        #self.dicts = {}
        for cname in self.char_cols:
            dicts[cname] = self.make_dict(self.df[cname].fillna(''))

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))

    def run(self, mode):
        global dicts
        print ("enter run mode " + str(mode))
        
        col_name = self.primary_field.split("|")[0]
        file_name = self.primary_field.split("|")[1]
        df_primary = self.pd.read_csv(workdir+file_name, encoding='utf8')[[col_name]]
        df_primary.merge
    
        for cname in self.char_cols:
            dict1 = dicts[cname]
            self.df[cname] = self.df[cname].fillna('').map(dict1)
            self.pd.DataFrame(list(dict1.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+cname+'.csv', encoding='utf-8')    #save new column dict
        
        self.df.to_csv(workdir+self.newfilename, index=False)
        
        nrow = len(self.df)

        for cname in self.df.columns:
            if cname in self.char_cols:
                is_dict="Y"
            else:
                is_dict="N"
            if cname==self.target:
                is_target="Y"
            else:
                is_target="N"
            print ("#add_field:"+cname+","+is_dict+","+self.newfilename+","+is_target+","+str(nrow))
    
    def apply(self, df_add):
        global dicts
        for index, row in df_add.iterrows():
            for cname in df_add.columns:
                if cname in self.char_cols:
                    if not (row[cname] in dicts[cname]):
                        dicts[cname][row[cname]] = 1+max(dicts[cname].values())
                    df_add.at[index, cname] = dicts[cname][row[cname]]
                else:
                    df_add.at[index, cname] = row[cname]

agent_{id} = cls_agent_{id}()
