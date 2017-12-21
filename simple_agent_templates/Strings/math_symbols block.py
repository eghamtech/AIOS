if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'msym_'
    symbols = '*+-/%#@^!'

    def __init__(self):
        global dicts
        
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.fldprefix = self.field_prefix + str(self.result_id)

    def run_on(self, df_run):
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        
        df1 = df_run[self.col1].map(self.dict1)
        idx = 0
        for c in self.symbols:
            idx+=1
            df_run.loc[:,self.fldprefix + '_' + str(idx)] = df1.apply(lambda x: (c in set(x))+0 if type(x)==str else 0)
    
    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.run_on(self.df)
        
        total_cols = len(self.symbols)
        nrow = len(self.df)
        
        for i in range(1,total_cols+1):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)
    
agent_{id} = cls_agent_{id}()
