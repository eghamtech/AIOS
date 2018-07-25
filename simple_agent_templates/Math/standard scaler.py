# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates 3 new columns by applying 3 methods of scaling to a selected field

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.preprocessing import Imputer
    from sklearn.externals import joblib
    from numpy import inf
    import os.path

    # obtain random field
    # restrict selection to those that not already used and not created by the agent
    col_definition1 = "{random_field_distinct}"   # use all columns including index columns of dict fields
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]

    # obtain a unique ID for the current instance
    result_id = {id}
    agent_name = 'agent_' + str(result_id)
    
    sfx = ["_ss_", "_mm_", "_qt_"]
    scalers = []
    col_max_min = {}
    
    def __init__(self):
        # if saved model already exists then load it from filesystem
        if self.os.path.isfile(workdir + self.agent_name + '_max_min.model'): 
            self.col_max_min = self.joblib.load(workdir + self.agent_name + '_max_min.model')
        
        if self.os.path.isfile(workdir + self.agent_name + '_imputer.model'): 
            self.imp = self.joblib.load(workdir + self.agent_name + '_imputer.model')
            
        for i in range(1,4):
            output_column = "scaled_" + self.col1 + self.sfx[i-1] + str(self.result_id)
            if self.os.path.isfile(workdir + output_column + '.model'):
                scaler = self.joblib.load(workdir + output_column + '.model')
                self.scalers.append(scaler)
            
            
    def run(self, mode):
        from numpy import inf
        print ("enter run mode " + str(mode))    
        # "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
        # read the data for selected column
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        nrow = len(self.df)
        
        if len(self.df[self.col1].unique()) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
            return
        
        # convert selected field to Numpy Array first and reshape as required by sklearn
        np_base_column = self.np.array(self.df[self.col1]).reshape(-1, 1)
        # find max and min values ignoring NaN and INF values
        np_column_max = self.np.ma.masked_invalid(np_base_column).max()
        np_column_min = self.np.ma.masked_invalid(np_base_column).min() 
        # replace INF and -INF with above
        np_base_column[np_base_column == inf] = np_column_max
        np_base_column[np_base_column == -inf] = np_column_min
        self.col_max_min = {'min':np_column_min, 'max':np_column_max}
        self.joblib.dump(self.col_max_min, workdir + self.agent_name + '_max_min.model')

        # replace NaN with mean values
        self.imp = self.Imputer()
        self.imp.fit(np_base_column)
        np_column = self.imp.transform(np_base_column)
        self.joblib.dump(self.imp, workdir + self.agent_name + '_imputer.model')
        
        #print( self.np.argwhere(self.np.isnan(self.np.array(df))) )
        #print( self.np.argwhere(self.np.isinf(self.np.array(df))) )

        for i in range(1,4):
            # create new field name with unique instance ID
            # and filename to save new field data
            output_column = "scaled_" + self.col1 + self.sfx[i-1] + str(self.result_id)
            output_filename = output_column + ".csv"
            # use sklearn library to scale col1 in df, pre-processed as np_column
            if i==1:
                scaler = self.StandardScaler()
            elif i==2:
                scaler = self.MinMaxScaler()
            elif i==3:
                scaler = self.QuantileTransformer(n_quantiles=100)
            
            try:
                self.df[output_column] = scaler.fit_transform(np_column)
            except:
                print ("Error applying Scaler " + str(i) + " ("+self.col1+")" + " setting new column to 0")
                self.df[output_column] = 0
                
            self.df[[output_column]].to_csv(workdir+output_filename)
        
            print ("Scaler " + str(i) + " ("+self.col1+")" + " saved to file")
            print ("#add_field:"+output_column+",N,"+output_filename+","+str(nrow))
                    
            scaler_filename = output_column + ".model"
            self.joblib.dump(scaler, workdir+scaler_filename)
    
    def apply(self, df_add):
        np_column = self.np.array(df_add[self.col1]).reshape(-1, 1)
        np_column[np_column == self.inf] = self.col_max_min.get('max',0)
        np_column[np_column == -self.inf] = self.col_max_min.get('min',0)
        try:
            np_column = self.imp.transform(np_column)
        except:
            print ("Error applying Imputer("+self.col1+")" + " ignoring it")

        for i in range(1,4):
            output_column = "scaled_" + self.col1 + self.sfx[i-1] + str(self.result_id)         
            try:
                df_add[output_column] = self.scalers[i-1].transform(np_column)
            except:
                print ("Error applying Scaler " + str(i) + " ("+self.col1+")" + " setting new column to 0")
                df_add[output_column] = 0


agent_{id} = cls_agent_{id}()
