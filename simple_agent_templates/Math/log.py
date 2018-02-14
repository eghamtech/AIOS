# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new column by apply natural logarithm function

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np

    # obtain random field of numerical type
    # selection is restricted to those that not already used and not created by the agent
    col_definition1 = "{random_field_numeric_distinct}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]

    # obtain a unique ID for the current instance
    result_id = {id}
    field_prefix = "field_math_"
    output_column = field_prefix + str(result_id)
    output_filename = output_column + ".csv"

    def run_on(self, df_run):
        df_run[self.output_column] = self.np.log(df_run[self.col1])

    def run(self, mode):
        # this is main method called by AIOS to process data
        # "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        self.run_on(self.df)
        
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        print ("log("+self.col1+")")
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(len(self.df)))
    
    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
