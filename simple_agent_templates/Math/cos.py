#no_permutation

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd
    import numpy as np

    col_definition1 = "{random_field_numeric}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]

    result_id = {id}
    field_prefix = "field_"
    output_column = field_prefix + str(result_id)
    output_filename = output_column + ".csv"

    def run_on(self, df_run):
        df_run[self.output_column] = self.np.cos(df_run[self.col1])

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        self.run_on(self.df)
        
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        print ("cos("+self.col1+")")
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(len(self.df)))
    
    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
