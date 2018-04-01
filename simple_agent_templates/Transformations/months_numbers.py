#start_of_parameters
#key=start_year;  type=constant;  value=enter_start_year
#key=year_column;  type=constant;  value=enter_year_column
#key=month_column;  type=constant;  value=enter_month_column
#key=output_column;  type=constant;  value=enter_output_column
#end_of_parameters
if 'dicts' not in globals():
    dicts = {}
    
class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np

    start_year = {start_year}
    col_year = "{year_column}"
    col_month = "{month_column}"
    filename = trainfile

    output_column = "{output_column}"
    output_filename = output_column + ".csv"

    def run_on(self, df_run):
        df_run[self.output_column] = (df_run[self.col_year]-self.start_year)*12 + df_run[self.col_month]
        

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.filename, usecols=[self.col_year, self.col_month])
        
        self.run_on(self.df)
        
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(len(self.df)))
    
    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
