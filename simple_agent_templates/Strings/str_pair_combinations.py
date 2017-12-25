if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    col_definition2 = "{random_dict}"
    col2 = col_definition2.split("|")[0]
    file2 = col_definition2.split("|")[1]
    result_id = {id}
    field_prefix = 'comb_'
    fldprefix = field_prefix + str(result_id)

    def __init__(self):
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        if self.col2 not in dicts:
            self.dict2 = self.pd.read_csv(workdir+'dict_'+self.col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict2 = {v:k for k,v in dicts[self.col2].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = self.df[self.col1].map(self.dict1)
        self.dfx[self.col2] = self.df[self.col2].map(self.dict2)
        
        block = int(len(self.dfx)/50)

        self.qcomb = {}
        k = block - 1
        for index, row in self.dfx.iterrows():
            k+=1
            if type(row[self.col1])==str:
                q1 = row[self.col1]
            else:
                q1 = ''
            if type(row[self.col2])==str:
                q2 = row[self.col2]
            else:
                q2 = ''

            if q1 in self.qcomb:
                self.qcomb[q1][q2] = 1
            else:
                self.qcomb[q1] = {q2 : 1}

            if q2 in self.qcomb:
                self.qcomb[q2][q1] = 1
            else:
                self.qcomb[q2] = {q1 : 1}

            if k >= block:
                print (index)
                k=0
        
        print ( "ok. qcomb loaded" )

    def run_on(self, df_run):
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        if self.col2 not in dicts:
            self.dict2 = self.pd.read_csv(workdir+'dict_'+self.col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict2 = {v:k for k,v in dicts[self.col2].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
        self.dfx[self.col2] = df_run[self.col2].map(self.dict2)
        
        block = int(len(self.dfx)/50)
        
        k = block-1
        for index, row in self.dfx.iterrows():
            k+=1
            if type(row[self.col1])==str:
                sline1 = row[self.col1]
            else:
                sline1 = ''
            if type(row[self.col2])==str:
                sline2 = row[self.col2]
            else:
                sline2 = ''

            if sline1 in self.qcomb:
                self.qcomb[sline1][sline2] = 1
            else:
                self.qcomb[sline1] = {sline2 : 1}

            if sline2 in self.qcomb:
                self.qcomb[sline2][sline1] = 1
            else:
                self.qcomb[sline2] = {sline1 : 1}

            res = 0
            res += len(self.qcomb[sline1])
            qs1 = set(self.qcomb[sline1].keys())

            res += len(self.qcomb[sline2])
            qs2 = set(self.qcomb[sline2].keys())

            res2 = len(qs1.intersection(qs2))

            df_run.set_value(index, self.fldprefix + '_1', res)
            df_run.set_value(index, self.fldprefix + '_2', res2)

            if k>=block and block>=1000:
                k=0
                print (index)

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.run_on(self.df)
        
        nrow = len(self.df)
        
        total_cols = 2

        for i in range(1,total_cols+1):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()


