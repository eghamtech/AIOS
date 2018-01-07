#start_of_parameters
#key=word_count_max;  type=constant;  value=enter_word_count_max
#end_of_parameters
if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'glv_'
    fldprefix = field_prefix + str(result_id)
    nwords = {word_count_max}
    error = 0


    def run_on(self, df_run):
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        
        cols = []
        for i in range(0,self.nwords*300):
            fld = self.fldprefix + '_' + str(i)
            cols.append(fld)
        dfx2 = self.pd.DataFrame(0, index=self.np.arange(len(df_run)), columns=cols)
        
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
        
        print ("start adding columns")
        df_run = df_run.join(dfx2)
        print ("ended adding columns")
        
        block = int(len(df_run)/10000)
        i = 0

        import requests
        import json
        
        for index, row in self.dfx.iterrows():
            i+=1
            if type(row[self.col1])==str:
                sline1 = row[self.col1]
            else:
                sline1 = ''
            
            #values = [0]*(self.nwords*300)
            r = requests.post("https://os.aichoo.ai:8080", verify=False, data={'action': 'glove_numbers', 'word_count_max': self.nwords, 'string': sline1})
            if r.status_code!=200:
                print(r.reason)
                print("#error")
                self.error = 1
                break
            
            obj = json.loads(r.text)
            values = [float(v) for v in obj['data'].split(',')]
            if len(values)!=self.nwords*300:
                print("wrong response length")
                print("#error")
                self.error = 1
                break
            
            #if index==0 and len(df_run)>1:
            #    print('creating columns: ')
            #    last = 0
            cnt = len(values)
            for j in range(0, cnt):
                #if index==0 and len(df_run)>1:
                #    last+=1
                #    if last>=10:
                #        print(str(j) + "/" + str(cnt) + "...")
                #        last = 0
                df_run.set_value(index, self.fldprefix + '_' + str(j), values[j])

            if i>=block and block>=10:
                i=0
                print (index)
    
    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        self.run_on(self.df)
        
        if self.error==1:
            return
        
        nrow = len(self.df)
        
        for i in range(0,self.nwords*300):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)
    
agent_{id} = cls_agent_{id}()
