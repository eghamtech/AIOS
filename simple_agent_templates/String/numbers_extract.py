#start_of_parameters
#key=max_numbers_to_extract;  type=constant;  value=5
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent extracts numerical values from a dict column and creates max_numbers_to_extract new columns


if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import re
        
    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'num_' + col1 + '_'
    fldprefix = field_prefix + str(result_id)
    
    n_new_fields = {max_numbers_to_extract}
    new_cols = []
    
    def __init__(self):
        # prepare list of n_new_fields new columns
        for i in range(0,self.n_new_fields):
            fld = self.fldprefix + '_' + str(i)
            self.new_cols.append(fld)
    
    def text_to_numbers(self, s):
       if type(s) == str:
          return self.re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
       else:
          return self.np.NaN
          
    def expand_list(self, l):
       n_e = {max_numbers_to_extract}
       if type(l) != list:
          return [0] * n_e
       elif len(l) >= n_e:
          return l[0:n_e]
       return [float(i) for i in l] + [0] * (n_e - len(l))
       
    
    def run_on(self, df_run): 
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
        
        self.dfx = self.pd.DataFrame(self.dfx[self.col1].apply(self.text_to_numbers)) # find all numbers
        self.dfx = self.dfx[self.col1].apply(self.expand_list)                        # bring all records (lists) to fixed size
        self.dfx = self.pd.DataFrame(self.dfx.values.tolist())                        # extract lists into df columns
        
   
    def run(self, mode):
        print ("enter run mode " + str(mode))
        df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
            return
        
        # prepare dataframe with n_new_fields new columns and init with 0.0 
        dfx2 = self.pd.DataFrame(0.0, index=self.np.arange(len(self.df)), columns=cols)
        
        print ("start adding columns")
        # join original column with new columns
        self.df = self.df.join(dfx2)
        print ("ended adding columns")
        
        self.run_on(df)
        
        nrow = len(df)

        df[[self.fldprefix + '_y']].to_csv(workdir+self.fldprefix + '_y.csv')
        print ("#add_field:"+self.fldprefix + '_y'+",N,"+self.fldprefix + '_y.csv'+","+str(nrow))
        df[[self.fldprefix + '_m']].to_csv(workdir+self.fldprefix + '_m.csv')
        print ("#add_field:"+self.fldprefix + '_m'+",N,"+self.fldprefix + '_m.csv'+","+str(nrow))
        df[[self.fldprefix + '_d']].to_csv(workdir+self.fldprefix + '_d.csv')
        print ("#add_field:"+self.fldprefix + '_d'+",N,"+self.fldprefix + '_d.csv'+","+str(nrow))
        df[[self.fldprefix + '_ts']].to_csv(workdir+self.fldprefix + '_ts.csv')
        print ("#add_field:"+self.fldprefix + '_ts'+",N,"+self.fldprefix + '_ts.csv'+","+str(nrow))
        
    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
