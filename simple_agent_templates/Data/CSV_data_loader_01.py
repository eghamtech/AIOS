#start_of_parameters
#key=source_filename;  type=constant;  value=enter_source_filename
#key=target;  type=constant;  value=enter_target
#end_of_parameters

# This script will scan your CSV file for string columns, convert them to dictionaries
# and create columns in AIOS Memory with data for each column in the 'source_filename' CSV file.
# Provide correct 'source_filename' in the agent parameters.
# Variables 'workdir' and 'trainfile' (target filename) must be setup in 'Constants' area of AIOS
# Parameter 'target' specifies column to be marked as the prediction target.

if 'dicts' not in globals():
    dicts = {}  # dict of dicts. each of dicts has structure: key=string, value=number

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import re, bz2, pickle, os.path
    
    source_filename = "{source_filename}"
    target = "{target}"
    # newfilename = trainfile
    # obtain a unique ID for the current instance
    result_id = {id}
    agent_name = 'csv_agent_' + str(result_id)
    colmap = {}
    char_cols = []
    
    def __init__(self):
        global dicts
        # if saved model for dictionaries already exists then load it from filesystem
        if self.os.path.isfile(workdir + self.agent_name + '.model'):
            rfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'r')
            dicts = self.pickle.load(rfile)
            rfile.close()
            
            self.colmap = dicts[self.agent_name + '.colmap']
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
        self.df = self.pd.read_csv(workdir+self.source_filename, encoding='utf8')
        
        new_cols = []
        self.colmap = {}
        for c in self.df.columns:
              str1 = c
              str1 = self.re.sub('[^0-9a-zA-Z]+', '_', str1)
              str1 = str1 + "_" + str(self.result_id)
              new_cols.append(str1)                              # list of new columns
              self.colmap[c] = str1                              # a map from old column names to new ones
        self.df.columns = new_cols                               # assign new column names to the dataframe
        
        print (str(datetime.now()), " processing TEXT columns")
        self.char_cols = list(self.df.select_dtypes(include=['object']).columns)
        print ("char columns: ", self.char_cols)
            
        for cname in self.char_cols:
            dict_char = self.make_dict(self.df[cname].fillna(''))
            dicts[cname] = dict_char
            self.df[cname] = self.df[cname].fillna('').map(dict_char)
            # save dictionary for each text column into separate file
            self.pd.DataFrame(list(dict_char.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+cname+'.csv', encoding='utf-8')
            print ("text column: " + cname + " processed")
               
        print (str(datetime.now()), " saving dicts...")           
        dicts[self.agent_name + '.colmap'] = self.colmap
        dicts[self.agent_name + '.char_cols'] = self.char_cols
       
        sfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        self.pickle.dump(dicts, sfile) 
        sfile.close()
        
        print (str(datetime.now()), " ...dicts saved.")
        # self.df.to_csv(workdir+self.newfilename, index=False) - deprecated
        
        nrow = len(self.df)
        is_use_for_models="Y"
        for cname in self.df.columns:
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
            output_column = cname
            output_filename = output_column + ".csv"
            self.df[[output_column]].to_csv(workdir+output_filename)           
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
                df_add[cshort] = float('nan')
            else:
                df_add.rename(index=str, columns={creal: cshort}, inplace=True)
                
        for index, row in df_add.iterrows():
            for cname in df_add.columns:
                if cname in self.char_cols:
                    if not (row[cname] in dicts[cname]):
                        dicts[cname][row[cname]] = 1+max(dicts[cname].values())
                    df_add.at[index, cname] = dicts[cname][row[cname]]
                #else:
                #    df_add.at[index, cname] = row[cname]

agent_{id} = cls_agent_{id}()
