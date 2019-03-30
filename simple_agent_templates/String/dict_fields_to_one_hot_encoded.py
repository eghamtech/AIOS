#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv','dict_field1|dict_field1.csv','dict_field2|dict_field2.csv']
#key=max_unique_values;  type=constant;  value=50
#key=col_max_length;  type=constant;  value=200
#key=new_field_prefix;  type=constant;  value=onehe_multiple_fields_
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns from given fields by hot encoding every unique value as 0 or 1
# all source fields expected to be dictionary fields
#
# number of new columns created will be the same as number of unique values across all source fields
# if it is no larger than "max_unique_values"
# each column name will be suffixed with a corresponding string value
# 
# if number of unique values exceed "max_unique_values" new columns will correspond to top "max_unique_values" most occurring 

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import os.path, bz2, pickle, re
    
    data_defs = {fields_source}
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
    max_unique_values = {max_unique_values}
    col_max_length    = {col_max_length}
    agent_name        = 'agent_' + str(result_id)

    dicts_agent = {}
    new_columns = []
    dict_cols   = []
    
    def is_set(self, s):
        return len(s)>0 and s!="0"
    
    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def __init__(self):
        from datetime import datetime
        # if saved dictionaries for the target field already exist then load them from filesystem              
        if self.os.path.isfile(workdir + self.agent_name + '.model'):
            rfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'r')
            self.dicts_agent = self.pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.agent_name + ': one hot encoding agent dictionaries model loaded')

            
    def run_on(self, df_run, apply_fun=False):
        self.new_columns = []
        
        if apply_fun:
            for col_name in self.dicts_agent['dict_cols']:           
                df_run['dict_'+col_name] = df_run[col_name].map( self.dicts_agent[col_name] )    

        for k,v in self.dicts_agent['all_dicts'].items():
            # all allowed values should be stored in this dictionary as keys, so just iterate over them
            new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_' + self.re.sub('[^0-9a-zA-Z]+', '_', str(k))
            new_col_name = new_col_name[:self.col_max_length]
            self.new_columns.append(new_col_name)
            df_run[new_col_name] = 0

        for index, row in df_run.iterrows():
            for col_name in self.dicts_agent['dict_cols']:
                
                value = row['dict_'+col_name]

                if self.dicts_agent['all_dicts'].get(value) != None:
                    new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_' + self.re.sub('[^0-9a-zA-Z]+', '_', str(value))
                    new_col_name = new_col_name[:self.col_max_length]
                    df_run.at[index, new_col_name] = 1
        
    
    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df_all   = self.pd.DataFrame(columns=['all_dict_columns'])
        
        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if i==0:
                self.df = self.pd.read_csv(workdir+file_name)[[col_name]]
            else:
                self.df = self.df.merge(self.pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)
                
            if self.os.path.isfile(workdir + 'dict_' + file_name):
                # load dictionary if it exists
                dict_temp = self.pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
                
                self.dicts_agent[col_name] = dict_temp               
                self.df['dict_'+col_name]  = self.df[col_name].map(dict_temp)
                
                self.dict_cols.append(col_name)
                self.df_all = self.df_all.append( self.df[['dict_'+col_name]].rename(index=str, columns={'dict_'+col_name: 'all_dict_columns'}), ignore_index=True )
                
        
        unique_list = self.df_all['all_dict_columns'].unique()
   
        if len(unique_list) == 1:
            print ("Selected columns contains only 1 unique value - no point to do anything with it.")
            return    
     
        df_top_values = self.df_all.groupby('all_dict_columns')['all_dict_columns'].agg({"count": len}).sort_values("count", ascending=False).head(self.max_unique_values).reset_index()
        dict_temp     = self.make_dict(df_top_values['all_dict_columns'].fillna('')) 
        
        self.dicts_agent['dict_cols'] = self.dict_cols 
        self.dicts_agent['all_dicts'] = dict_temp
               
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
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

        
    def apply(self, df_add):
        self.run_on(df_add, apply_fun=True)
        
agent_{id} = cls_agent_{id}()
