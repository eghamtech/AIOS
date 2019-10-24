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
# Variable 'workdir' must be setup in 'Constants' area of AIOS
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
    import re, bz2, pickle, os.path
    
    source_filename      = "{source_filename}"             # file in workdir where additional data is
    source_primary_field = "{source_primary_field}"        # name of the field which can be used to link to existing data
    primary_field    = "{primary_field}"                   # name of the link field in the existing data
    new_field_prefix = "{field_prefix}"
    target = "{target}"
   
    # obtain a unique ID for the current instance
    result_id  = {id}
    agent_name = 'csv_02_agent_' + str(result_id)
    
    colmap    = {}
    char_cols = []
    
    def printlog(self, mesg):
        from datetime import datetime
        global DEBUG
        if DEBUG == 1:
            print (str(datetime.now()), mesg)
    
    def clean_text(self, s):
        # Replace symbols with language
        s = s.replace('&', '_and_')
        s = s.replace('#', '_sharp_')
        s = s.replace('@', '_at_')
        s = s.replace('*', '_star_')
        s = s.replace('%', '_prcnt_')
        
        s = s.replace('(', '_ob_')
        s = s.replace(')', '_cb_')
        s = s.replace('{', '_ocb_')
        s = s.replace('}', '_ccb_')
        s = s.replace('[', '_osb_')
        s = s.replace(']', '_csb_')
        
        s = s.replace('=', '_eq_')
        s = s.replace('>', '_gt_')
        s = s.replace('<', '_lt_')
        s = s.replace('+', '_plus_')
        s = s.replace('-', '_dash_')
        s = s.replace('/', '_fsl_')
        s = s.replace('\\', '_bsl_')
        s = s.replace('?', '_qm_')
        s = s.replace('!', '_em_')

        s = s.replace('.', '_dot_')
        s = s.replace(',', '_coma_')
        s = s.replace(':', '_cln_')
        s = s.replace(';', '_scln_')

        s = self.re.sub('[^0-9a-zA-Z]+', '_', s)
        return  s
        
    def __init__(self):
        global dicts
        # if saved model for dictionaries already exists then load it from filesystem
        if self.os.path.isfile(workdir + self.agent_name + '.model'):
            rfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'r')
            dicts = self.pickle.load(rfile)
            rfile.close()
            
            self.colmap    = dicts[self.agent_name + '.colmap']
            self.char_cols = dicts[self.agent_name + '.char_cols']

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))

    def run(self, mode):
        from datetime import datetime
        global dicts
        print ("enter run mode " + str(mode))
        print (str(datetime.now()), " creating dataframe...")
        self.df = self.pd.read_csv(workdir+self.source_filename, encoding='utf8', engine='python', error_bad_lines=False)
        
        # rename all new columns by removing non alfa-numeric symbols
        cols            = self.df.columns
        new_cols        = []                                     # list of new columns (with possible duplicates)
        new_cols_unique = []                                     # list of new columns (fully unique names)
        self.colmap     = {}
        
        for i in range(0, len(cols)):
            str1 = self.new_field_prefix + "_" + cols[i]
            str1 = self.clean_text(str1)
            str1 = str1[:220]                                   # limit column name length to comply with Linux filename limit                 
            new_cols.append(str1)                               # list of new columns
              
            ncol_count = new_cols.count(str1)
            if ncol_count==1:
                str1 = str1 + "_" + str(self.result_id)              
            else:
                str1 = str1 + "_dpl" + str(ncol_count) + "_" + str(self.result_id)
            
            new_cols_unique.append(str1)          
            self.colmap[cols[i]] = str1                          # a map from old column names to new ones
        
        self.df.columns = new_cols_unique                        # assign new column names to the dataframe

        col_name  = self.primary_field.split("|")[0]
        file_name = self.primary_field.split("|")[1]
        
        # create a copy of specified primary field to have the same name as primary field in AIOS Memory
        # so dataframe can be joined on that field
        self.df[col_name] = self.df[self.colmap[self.source_primary_field]]
        
        df_primary = self.pd.read_csv(workdir+file_name, encoding='utf8')[[col_name]]
        
        df_primary = self.pd.merge(df_primary, self.df, how='left', on=col_name, sort=False)
        df_primary.drop(col_name, axis=1, inplace=True)              # remove primary field as it is a duplicate previously created
        
        print (str(datetime.now()), " processing TEXT columns")
        self.char_cols = list(df_primary.select_dtypes(include=['object']).columns)
        print ("source data loaded")
        print ("rows: ", len(df_primary))
 
        for cname in self.char_cols:
            dict_char         = self.make_dict(df_primary[cname].fillna(''))
            dicts[cname]      = dict_char
            df_primary[cname] = df_primary[cname].fillna('').map(dict_char)
            # save dictionary for each text column into separate file
            self.pd.DataFrame(list(dict_char.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+cname+'.csv', encoding='utf-8')
            print ("text column: " + cname + " processed")
        
        print (str(datetime.now()), " saving dicts...")           
        dicts[self.agent_name + '.colmap']    = self.colmap
        dicts[self.agent_name + '.char_cols'] = self.char_cols
       
        sfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        self.pickle.dump(dicts, sfile) 
        sfile.close()
        
        print (str(datetime.now()), " ...dicts saved.")
        
        nrow = len(df_primary)
        is_use_for_models="Y"
        for cname in df_primary.columns:
            if cname in self.char_cols:
                is_dict="Y"
            else:
                is_dict="N"
            if cname==self.target:
                is_target="Y"
                is_use_for_models="N"
            else:
                is_target="N"
                
            # save each column into separate file and register new field
            output_column   = cname
            output_filename = output_column + ".csv"
            df_primary[[output_column]].to_csv(workdir+output_filename)           
            print ("#add_field:"+output_column+","+is_dict+","+output_filename+","+is_target+","+str(nrow)+","+is_use_for_models)
           
    
    def apply(self, df_add):
       # this function is called when new data needs to be processed
        # new data supplied in df_add
        global dicts
        
        # creal - original column name; cshort - converted column name
        # check which of the fields originally processed by this class present in df_add and 
        # create missing ones with NaN or rename as previously done
        for creal, cshort in self.colmap.items():
            if creal not in df_add.columns:
                self.printlog ("CSV Loader: column <" + creal + "> missing! Created with NaN")
                df_add[cshort] = float('nan')
            else:
                df_add.rename(index=str, columns={creal: cshort}, inplace=True)
                
            if (cshort in self.char_cols): 
                df_add[cshort].fillna('', inplace=True)
        self.printlog ("CSV Loader: columns renamed")
                
        for index, row in df_add.iterrows():                                      # iterate over each row in df_add 
            for cname in df_add.columns:                                          # iterate over each column in df_add row
                if (cname in self.char_cols):
                    cname_dict = dicts[cname]         
                    cname_value = str(row[cname]) if row[cname] != None else ''
                    self.printlog ("CSV Loader: text column " + cname + "; value: " + cname_value)
                    
                    if not (cname_value in cname_dict):                           # if value in current row and column not in dictionary
                        self.printlog ("CSV Loader: column " + cname + "; value: " + cname_value + " not in dictionary")
                        new_key = 1 + max(cname_dict.values())                    # create new key with max+1 value
                        dicts[cname][cname_value] = new_key                       # add text:key to original dictionary
                        df_add.at[index, cname] = new_key
                    else:    
                        df_add.at[index, cname] = cname_dict[cname_value]
                    
                    self.printlog ("CSV Loader: column " + cname + "; value: " + cname_value + " mapped")

agent_{id} = cls_agent_{id}()
