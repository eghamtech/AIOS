# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent analyses selected field and removes it from being used if it has no useful information

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import numpy as np
    
    # obtain random field of any type
    col_definition1 = "{random_field_distinct}"   # use all columns including index columns of dict fields
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]

    # obtain a unique ID for the current instance
    result_id = {id}

    # "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
    # read the data for selected column
    df = pd.read_csv(workdir+file1)[[col1]]
    
    def run(self, mode):
        print ("enter run mode " + str(mode))
        nrow = len(self.df)
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
        else:
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",Y")

        
    def apply(self, df_add):
       return

agent_{id} = cls_agent_{id}()
