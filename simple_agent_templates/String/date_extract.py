# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent extracts names of months and numerical days from a dict column, attempts to recognise if this is a date 
# and converts to 4 new fields: Y M D TS

if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import re
    import datetime
    import time
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    months2 = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        
    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'dte_' + col1 + '_'
    fldprefix = field_prefix + str(result_id)
    
    
    def get_month(self, x):
        if self.pd.isnull(x):
            return x
        else:
            x = x.split(" ")
            for item in x:
                item = item.lower()
                if item in self.months:
                    return self.months.index(item)+1
                elif item in self.months2:
                    return self.months2.index(item)+1
            return -1

    def get_day(self, x):
        if self.pd.isnull(x):
            return x
        else:
            x = x.split(" ")
            for item in x:
                if len(item)>0:
                    if item[0]!="'":
                        item = int("0"+self.re.sub("[^0-9]", "", item))
                        if item>0 and item <= 31:
                            return item
            return -1

    def get_year(self, x):
        if self.pd.isnull(x):
            return x
        else:
            x = x.split(" ")
            for item in x:
                if len(item)>0:
                    item_n = int("0"+self.re.sub("[^0-9]", "", item))
                    if item[0]=="'":
                        if item_n>30:
                            return 1900+item_n
                        elif item_n>0:
                            return 2000+item_n
                    else:
                        if item_n>1900 and item_n<2100:
                            return item_n
            return -1
    
    def get_ts(self, row):
        if row[0]>0 and row[1]>0 and row[2]>0:
            try:
                dt = self.datetime.datetime(year=int(row[0]), month=int(row[1]), day=int(row[2]))
                return self.time.mktime(dt.timetuple())
            except ValueError:
                return float('nan')
        else:
            return float('nan')
    
    def run_on(self, df_run):
        
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
        
        df_run[self.fldprefix + '_y'] = self.dfx[self.col1].apply(lambda x: self.get_year(x))
        df_run[self.fldprefix + '_m'] = self.dfx[self.col1].apply(lambda x: self.get_month(x))
        df_run[self.fldprefix + '_d'] = self.dfx[self.col1].apply(lambda x: self.get_day(x))
        df_run[self.fldprefix + '_ts'] = df_run[[self.fldprefix + '_y', self.fldprefix + '_m', self.fldprefix + '_d']].apply(self.get_ts, axis=1)

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
            return    
        
        self.run_on(self.df)
        
        nrow = len(self.df)

        self.df[[self.fldprefix + '_y']].to_csv(workdir+self.fldprefix + '_y.csv')
        print ("#add_field:"+self.fldprefix + '_y'+",N,"+self.fldprefix + '_y.csv'+","+str(nrow))
        self.df[[self.fldprefix + '_m']].to_csv(workdir+self.fldprefix + '_m.csv')
        print ("#add_field:"+self.fldprefix + '_m'+",N,"+self.fldprefix + '_m.csv'+","+str(nrow))
        self.df[[self.fldprefix + '_d']].to_csv(workdir+self.fldprefix + '_d.csv')
        print ("#add_field:"+self.fldprefix + '_d'+",N,"+self.fldprefix + '_d.csv'+","+str(nrow))
        self.df[[self.fldprefix + '_ts']].to_csv(workdir+self.fldprefix + '_ts.csv')
        print ("#add_field:"+self.fldprefix + '_ts'+",N,"+self.fldprefix + '_ts.csv'+","+str(nrow))
        
    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
