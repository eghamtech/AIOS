#start_of_parameters
#key=max_unique_values;  type=constant;  value=50
#key=col_max_length;  type=constant;  value=200
#key=new_field_prefix;  type=constant;  value=onehe_
#end_of_parameters

# AICHOO OS Simple Agent 
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns from given field by hot encoding every unique value as 0 or 1
# for dictionary fields, its dictionary will be loaded and used for column names
# number of new columns created will be the same as number of unique values
# each column name will be suffixed with a corresponding string value or value from dictionary of field is a dict field
# column values will be rounded to integer and if number of unique values exceed "max_unique_values" such column will be ignored

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import os.path
    
    col_definition1 = "{random_field_distinct}"
    col1  = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    
    # obtain a unique ID for the current instance
    result_id = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
	max_unique_values = {max_unique_values}
	col_max_length    = {col_max_length}
	
	dicts_agent = {}
	new_columns = []
	
    def is_set(self, s):
        return len(s)>0 and s!="0"
    
    def __init__(self):
		from datetime import datetime
        # if saved dictionaries for the target field already exist then load them from filesystem              
        if self.os.path.isfile(workdir + self.col1 + '_dicts.model'):
            rfile = self.bz2.BZ2File(workdir + self.col1 + '_dicts.model', 'r')
            self.dicts_agent = self.pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.col1 + ': one hot encoding agent dictionaries model loaded')

    def run_on(self, df_run):
		col_name = self.col1
		self.new_columns = []
		
		for k,v in self.dicts_agent[col_name].items():
			new_col_name = self.new_field_prefix + col_name + '_' + str(self.result_id) + '_v_' + str(v)
			new_col_name = new_col_name[:self.col_max_length]
			self.new_columns.append(new_col_name)
			df_run[new_col_name] = 0
		
		for index, row in df_run.iterrows():
			value = int(round(row[col_name],0))
			value_mapped = self.dicts_agent[col_name].get(value)
			
			if value_mapped != None:
				new_col_name = self.new_field_prefix + col_name + '_' + str(self.result_id) + '_v_' + str(value_mapped)
				new_col_name = new_col_name[:self.col_max_length]
				df_run.at[index, new_col_name] = 1
		
	
    def run(self, mode):
        print ("enter run mode " + str(mode))
		self.df  = self.pd.read_csv(workdir+self.file1)[[self.col1]]

		self.df[self.col1] = self.df[self.col1].apply(lambda x: int(round(x,0)))
		unique_list = self.df[self.col1].unique()
		
		if len(unique_list) == 1:
            print ("Selected column contains only 1 unique value - no point to do anything with it.")
            # register the same field as the source field, which notifies AIOS of successful exit
            # and instructs to mark such field with use_for_models=False
            print ("#add_field:"+self.col1+",N,"+self.file1+","+str(len(self.df))+",N")   
            return    
		
		if len(unique_list) > self.max_unique_values:
			print ("Selected column contains " + str(unique_v) + " unique values which is more than " + str(max_unique_values) + " allowed - ignoring it.")
			return
			
		file_name  = self.file1
		col_name   = self.col1
		# load dictionary if it exists
        if self.os.path.isfile(workdir + 'dict_' + file_name):
            dict_temp = self.pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
		else:
			dict_temp = {x:str(x) for x in unique_list if str(x) != 'nan'}
		
		self.dicts_agent[col_name] = dict_temp
        self.df["dict_"+col_name]  = self.df[col_name].map(dict_temp)
        
		self.run_on(self.df)                      
        nrow = len(self.df)
		
		self.dicts_agent['new_columns'] = self.new_columns
		# save dictionary of all auxiliary data into file
        sfile = self.bz2.BZ2File(workdir + col_name + '_dicts.model', 'w')
        self.pickle.dump(self.dicts_agent, sfile) 
        sfile.close()
		
		# save and register each new column
		for i in range(0,self.new_columns):
            fld   = self.new_columns[i]
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))
        
		
    def apply(self, df_add):
        self.run_on(df_add)
       

agent_{id} = cls_agent_{id}()
