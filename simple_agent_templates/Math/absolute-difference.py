#no_permutation

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns by substracting one from another in absolute terms

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd

    # obtain random field of numerical type
    # selection is restricted to those that not already used and not created by the agent
    # as well as avoiding permutation of the same two columns
    col_definition1 = "{random_field_numeric_distinct}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    # obtain second random field of numerical type
    col_definition2 = "{random_field_numeric_distinct}"
    col2 = col_definition2.split("|")[0]
    file2 = col_definition2.split("|")[1]

    # obtain a unique ID for the current instance
    result_id = {id}
    field_prefix = "field_math_"
    output_column = field_prefix + str(result_id)
    output_filename = output_column + ".csv"

    # "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
    # read the data for selected columns and merge into dataframe
    df = pd.read_csv(workdir+file1)[[col1]]
    df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)
        
    def run_on(self, df_run):
        df_run[self.output_column] = abs(df_run[self.col1] - df_run[self.col2])

    def run(self, mode):
        # this is main method called by AIOS to process data
        print ("enter run mode " + str(mode))
     
        self.run_on(self.df)
        
        self.df[[self.output_column]].to_csv(workdir+self.output_filename)
        print ("ABS(" + self.col1+" - "+self.col2+ ")")
        print ("#add_field:"+self.output_column+",N,"+self.output_filename+","+str(len(self.df)))
    
    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
