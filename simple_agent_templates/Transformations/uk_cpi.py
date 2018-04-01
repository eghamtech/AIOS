#start_of_parameters
#key=start_year;  type=constant;  value=enter_start_year
#key=month_number_column;  type=constant;  value=enter_month_number_column
#key=output_column;  type=constant;  value=enter_output_column
#end_of_parameters
#
#https://fred.stlouisfed.org/series/GBRCPIALLMINMEI
#GBRCPIALLMINMEI.csv

if 'dicts' not in globals():
    dicts = {}
    
class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import datetime

    start_year = {start_year}
    month_number_column = "{month_number_column}"

    output_column = "{output_column}"
    output_filename = output_column + ".csv"
    
    cpi_monthly = {}

    def __init__(self):
        cpim = self.pd.read_csv("yf1/GBRCPIALLMINMEI.csv")
        cpim["month_no"] = cpim["DATE"].apply(lambda x: (self.datetime.datetime.strptime(x, "%Y-%m-%d").year-{start_year})*12+self.datetime.datetime.strptime(x, "%Y-%m-%d").month)
        self.cpi_monthly = dict(zip(cpim.month_no, cpim.GBRCPIALLMINMEI))
        
    def run_on(self, df_run):
        df_run[self.output_column] = df_run[self.month_number_column].map(self.cpi_monthly)
        

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.month_number_column + ".csv", usecols=[self.month_number_column])
        
        self.run_on(self.df)
        
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(len(self.df)))
    
    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
