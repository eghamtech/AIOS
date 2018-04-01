#start_of_parameters
#key=target_cpi;  type=constant;  value=enter_target_cpi
#key=price_col;  type=constant;  value=enter_price_col
#key=cpi_col;  type=constant;  value=enter_cpi_col
#key=output_column;  type=constant;  value=enter_output_column
#end_of_parameters
if 'dicts' not in globals():
    dicts = {}
    
class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np
    import datetime

    price_col = "{price_col}"
    cpi_col = "{cpi_col}"

    output_column = "{output_column}"
    output_filename = output_column + ".csv"
    
    target_cpi = {target_cpi}
    
    def run_on(self, df_run):
        df_run[self.output_column] = self.np.log(self.np.exp(df_run[self.price_col]) / df_run[self.cpi_col] * self.target_cpi)
        

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+trainfile, usecols=[self.price_col])
        self.df = self.df.merge(self.pd.read_csv(workdir+self.cpi_col+'.csv', usecols=[self.cpi_col]), left_index=True, right_index=True)
        
        self.run_on(self.df)
        
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+",Y,"+str(len(self.df)))
    
    def apply(self, df_add):
        if self.price_col not in df_add.columns:
            df_add[self.price_col] = float('nan')
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()