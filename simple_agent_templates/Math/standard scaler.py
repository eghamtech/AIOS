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
    from numpy import inf

    # obtain random field of numerical type
    # restrict selection to those that not already used and not created by the agent
    col_definition1 = "{random_field_numeric_distinct}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]

    # obtain a unique ID for the current instance
    result_id = {id}
    
    sfx = ["_ss_", "_mm_", "_qt_"]
    scalers_loaded = False
    scalers = []
    
    # "workdir" must be specified in Constants - it is a global setting where all CSV files are stored on Jupyter server
    # read the data for selected column
    df = pd.read_csv(workdir+file1)[[col1]]
    # convert selected field to Numpy Array first and reshape as required by sklearn
    np_base_column = np.array(df[col1]).reshape(-1, 1)
    # find max and min values ignoring NaN and INF values
    np_column_max = np.ma.masked_invalid(np_base_column).max()
    np_column_min = np.ma.masked_invalid(np_base_column).min() 
    # replace INF and -INF with above
    np_base_column[np_base_column == inf] = np_column_max
    np_base_column[np_base_column == -inf] = np_column_min

    # replace NaN with mean values
    imp = Imputer()
    imp.fit(np_base_column)

    def run(self, mode):
        print ("enter run mode " + str(mode))
        
        nrow = len(self.df)

        #print( self.np.argwhere(self.np.isnan(self.np.array(df))) )
        #print( self.np.argwhere(self.np.isinf(self.np.array(df))) )

        # replace NaN with mean values
        np_column = self.imp.transform(self.np_base_column)

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
                print ("Error applying Scaler" + str(i) + "("+self.col1+")" + " setting new column to 0")
                self.df[output_column] = 0
                
            self.df[[output_column]].to_csv(workdir+output_filename)
        
            print ("Scaler" + str(i) + "("+self.col1+")" + " saved to file")
            print ("#add_field:"+output_column+",N,"+output_filename+","+str(nrow))
            
            from sklearn.externals import joblib
            scaler_filename = output_column + "_model.save"
            joblib.dump(scaler, workdir+scaler_filename)
    
    def apply(self, df_add):
        np_column = self.np.array(df_add[self.col1]).reshape(-1, 1)
        np_column[np_column == self.inf] = self.np_column_max
        np_column[np_column == -self.inf] = self.np_column_min
        np_column = self.imp.transform(np_column)

        for i in range(1,4):
            output_column = "scaled_" + self.col1 + self.sfx[i-1] + str(self.result_id)
            
            if not self.scalers_loaded:
                from sklearn.externals import joblib
                scaler_filename = output_column + "_model.save"
                scaler = joblib.load(workdir+scaler_filename)
                self.scalers.append(scaler)
            else:
                #use loaded scaler
                scaler = self.scalers[i-1]
            
            try:
                df_add[output_column] = scaler.transform(np_column)
            except:
                print ("Error applying Scaler" + str(i) + "("+self.col1+")" + " setting new column to 0")
                df_add[output_column] = 0
                
        self.scalers_loaded = True


agent_{id} = cls_agent_{id}()
