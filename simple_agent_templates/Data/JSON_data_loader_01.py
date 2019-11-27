#start_of_parameters
#key=source_filename_json;  type=constant;  value=enter_source_filename_json
#end_of_parameters

# Processes JSON file which has "training_data" and "model_definition" objects according to below specification in Wiki:
# https://github.com/eghamtech/AIOS/wiki/Input-data-JSON-format-01
# latest version saves each column into separate file inline with other agents
# it also saves dictionaries into model file, so agent can be applied without original data source

if 'dicts' not in globals():
    dicts = {}
    
class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import json, datetime, calendar, dateutil.parser
    import re, bz2, pickle, os.path
    
    source_filename = "{source_filename_json}"
    # newfilename = trainfile
    # obtain a unique ID for the current instance
    result_id  = {id}
    agent_name = 'agent_' + str(result_id)
    colmap      = {}
    char_cols   = []
    date_cols   = []
    lookup_cols = []

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def reverse_dict_int_key(self, dt): 
        return {v:int(k) for k,v in dt.items()}
    
    def printlog(self, mesg):
        from datetime import datetime
        global DEBUG
        if DEBUG == 1:
            print (str(datetime.now()), mesg)
    
    def clean_text(self, s):
        # Replace symbols with language
        s = s.replace('&', '_and_')
        s = s.replace('#', '_sharp_')
        s = s.replace('@', '_at_')
        s = s.replace('*', '_star_')
        s = s.replace('%', '_prcnt_')
        
        s = s.replace('(', '_ob_')
        s = s.replace(')', '_cb_')
        s = s.replace('{', '_ocb_')
        s = s.replace('}', '_ccb_')
        s = s.replace('[', '_osb_')
        s = s.replace(']', '_csb_')
        
        s = s.replace('=', '_eq_')
        s = s.replace('>', '_gt_')
        s = s.replace('<', '_lt_')
        s = s.replace('+', '_plus_')
        s = s.replace('-', '_dash_')
        s = s.replace('/', '_fsl_')
        #s = s.replace('\\', '_bsl_')
        s = s.replace('?', '_qm_')
        s = s.replace('!', '_em_')

        s = s.replace('.', '_dot_')
        s = s.replace(',', '_coma_')
        s = s.replace(':', '_cln_')
        s = s.replace(';', '_scln_')

        s = self.re.sub('[^0-9a-zA-Z]+', '_', s)
        return  s
    
    def __init__(self):
        global dicts
        
        # if saved model for dictionaries already exists then load it from filesystem
        if self.os.path.isfile(workdir + self.agent_name + '.model'):
            rfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'r')
            dicts = self.pickle.load(rfile)
            rfile.close()
            
            self.colmap      = dicts[self.agent_name + '.colmap']
            self.char_cols   = dicts[self.agent_name + '.char_cols']
            self.date_cols   = dicts[self.agent_name + '.date_cols']
            self.lookup_cols = dicts[self.agent_name + '.lookup_cols']
  
         
    def run(self, mode):
        from datetime import datetime
        
        global dicts
        print ("enter run mode " + str(mode))
        
        print (str(datetime.now()), " loading json file to dataframe...")
        
        with open(workdir + self.source_filename, encoding='utf-8') as f1:
            json_data = self.json.load(f1)
            f1.close()
        
        print (str(datetime.now()), " creating dataframe...")       
        self.df = self.pd.DataFrame().from_dict(json_data["training_data"])
        
        # identify columns based on their source of origin
        colmap_origin = {}
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item   = json_data["model_definition"]["layout"]["columns"][i]
            cname  = item["heading"]
            origin = item["origin"]
            colmap_origin[cname] = origin
            
        # rename all columns by removing non alfa-numeric symbols
        cols = self.df.columns
        new_cols        = []                                     # list of new columns (with possible duplicates)
        new_cols_unique = []                                     # list of new columns (fully unique names)
        self.colmap     = {}                                     # dictionary 'original_column' : 'new_column'
        for i in range(0, len(cols)):
            str1 = cols[i]
            # prefix field depending on its source
            #if colmap_origin[str1] == 'LL':
            #    str1 = 'orgnLL_' + str1
            
            str1 = self.clean_text(str1)
            str1 = str1[:220]                                    # limit column name length to comply with Linux filename limit 
            new_cols.append(str1)                           
            
            ncol_count = new_cols.count(str1)
            if ncol_count==1:
                str1 = str1 + "_" + str(self.result_id)              
            else:
                str1 = str1 + "_dpl" + str(ncol_count) + "_" + str(self.result_id)
            
            new_cols_unique.append(str1)          
            self.colmap[cols[i]] = str1                          # a map from old column names to new ones
        
        self.df.columns = new_cols_unique                        # assign new column names to the dataframe
        
        print (str(datetime.now()), " processing DATETIME columns...")
        self.date_cols = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item  = json_data["model_definition"]["layout"]["columns"][i]              # get column definition item
            cname = self.colmap[item["heading"]]                                       # obtain new name for the given column
            
            if item["data_type"]=='DATETIME':      
                if cname not in self.date_cols:
                    self.date_cols.append(cname)
                    print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                    self.df[cname+'_Y']  = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
                    self.df[cname+'_M']  = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
                    self.df[cname+'_D']  = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
                    self.df[cname+'_WD'] = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
                    self.df[cname+'_TS'] = self.df[cname].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
                    self.df = self.df.drop(cname, 1)
        
        print (str(datetime.now()), " processing FREETEXT/LARGETEXT columns")
        self.char_cols = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item  = json_data["model_definition"]["layout"]["columns"][i]              # get column definition item
            cname = self.colmap[item["heading"]]                                       # obtain new name for the given column
            
            if item["data_type"]=='FREETEXT' or item["data_type"]=='LARGETEXT':
                if cname not in self.char_cols:
                    print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                    self.char_cols.append(cname)
                    
                    dict_char = self.make_dict(self.df[cname].fillna(''))                  # create dictionary of given text column  
                    dicts[cname] = dict_char                                               # add current dictionary to the global dictionary of dictionaries
                    self.df[cname] = self.df[cname].fillna('').map(dict_char)              # replace column values with corresponding values from dictionary
                    # save dictionary for each text column into separate file
                    self.pd.DataFrame(list(dict_char.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+cname+'.csv', encoding='utf-8')
        # print ("char columns:", self.char_cols)
        
        print (str(datetime.now()), " processing LOOKUP columns")
        self.lookup_cols = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item  = json_data["model_definition"]["layout"]["columns"][i]              # get column definition item
            cname = self.colmap[item["heading"]]                                       # obtain new name for the given column
            
            if item["data_type"]=='LOOKUP':
                if cname not in self.lookup_cols:
                    print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"], "---", item["meta"]["lookup_id"], "---", item["meta"]["lookup_type"])
                    self.lookup_cols.append(cname)
                    
                    dict_lookup = {}
                    lookup_id   = str(item["meta"]["lookup_id"])
                    lookup_type = item["meta"]["lookup_type"]
                    for key in json_data["static_data"]["lookups"][lookup_type][lookup_id]["lookup_values"].keys():
                        dict_lookup[key] = json_data["static_data"]["lookups"][lookup_type][lookup_id]["lookup_values"][key]["value"]
                    
                    dicts[cname] = dict_lookup
                    self.pd.DataFrame(list(dict_lookup.items()), columns=['key', 'value']).to_csv(workdir+'dict_'+self.colmap[item["heading"]]+'.csv', encoding='utf-8')    #save new column dict
        # print ("lookup columns:", self.lookup_cols)
         
        print (str(datetime.now()), " processing DATA/OUTCOME columns")
        self.target_cols    = []
        self.use_for_models = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            cname = self.colmap[item["heading"]]                                       # obtain new name for the given column
            
            if item["analysis"]=='outcome':
                print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                self.target_cols.append(cname)
                self.df[cname] = self.df[cname].astype(int)
            elif item["analysis"]=='data':
                self.use_for_models.append(cname)
                if cname in self.date_cols:
                    self.use_for_models.append(cname+'_Y')
                    self.use_for_models.append(cname+'_M')
                    self.use_for_models.append(cname+'_D')
                    self.use_for_models.append(cname+'_WD')
                    self.use_for_models.append(cname+'_TS')
                
        print ("target columns:", self.target_cols)
        
        from numpy import nan
        self.df.fillna(value=nan, inplace=True)
        
        print (str(datetime.now()), " saving dicts...")           
        dicts[self.agent_name + '.colmap']      = self.colmap
        dicts[self.agent_name + '.char_cols']   = self.char_cols
        dicts[self.agent_name + '.date_cols']   = self.date_cols
        dicts[self.agent_name + '.lookup_cols'] = self.lookup_cols
       
        sfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        self.pickle.dump(dicts, sfile) 
        sfile.close()
        
        print (str(datetime.now()), " ...dicts saved.")
        # self.df.to_csv(workdir+self.newfilename, index=False)      # save all numeric columns into one csv file - deprecated
        
        nrow = len(self.df)

        for cname in self.df.columns:
            if (cname in self.char_cols) or (cname in self.lookup_cols):
                is_dict="Y"
            else:
                is_dict="N"
            if cname in self.target_cols:
                is_target="Y"
                is_use_for_models="N"
            else:
                is_target="N"
            if cname in self.use_for_models:
                is_use_for_models="Y"
            else:
                is_use_for_models="N"
            
            # save each column into separate file
            output_column   = cname
            output_filename = output_column + ".csv"
            self.df[[output_column]].to_csv(workdir+output_filename)
            
            # print ("#add_field:"+cname+","+is_dict+","+self.newfilename+","+is_target+","+str(nrow)+","+is_use_for_models)
            print ("#add_field:"+output_column+","+is_dict+","+output_filename+","+is_target+","+str(nrow)+","+is_use_for_models)
    
    
    def apply(self, df_add):
        # this function is called when new data needs to be processed
        # new data supplied in df_add
        global dicts
              
        # creal - original column name; cshort - converted column name
        # check which of the fields originally processed by this class present in df_add and 
        # create missing ones with NaN or rename as previously done
        for creal, cshort in self.colmap.items():
            if creal not in df_add.columns:
                self.printlog ("JSON Loader: column <" + creal + "> missing! Created with NaN")
                df_add[cshort] = float('nan')
            else:
                df_add.rename(index=str, columns={creal: cshort}, inplace=True)
                
            if (cshort in self.char_cols): 
                df_add[cshort].fillna('', inplace=True)
                
        self.printlog ("JSON Loader: columns renamed")
        
        for cname in self.date_cols:
            df_add[cname+'_Y']  = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_M']  = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_D']  = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_WD'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_TS'] = df_add[cname].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
            df_add.drop(cname, axis=1, inplace=True)
        self.printlog ("JSON Loader: date columns processed")
        
        for index, row in df_add.iterrows():                                      # iterate over each row in df_add 
            for cname in df_add.columns:                                          # iterate over each column in df_add row
                if (cname in self.char_cols):
                    cname_dict  = dicts[cname]         
                    cname_value = str(row[cname]) if row[cname] != None else ''
                    self.printlog ("JSON Loader: text column " + cname + "; value: " + cname_value)
                    
                    if not (cname_value in cname_dict):                           # if value in current row and column not in dictionary
                        self.printlog ("JSON Loader: column " + cname + "; value: " + cname_value + " not in dictionary")
                        new_key = 1 + max(cname_dict.values())                    # create new key with max+1 value
                        dicts[cname][cname_value] = new_key                       # add text:key to original dictionary
                        df_add.at[index, cname]   = new_key
                        
                        # since global dictionary is modified it needs to be saved to avoid issues with future predictions
                        # of columns with text not in dictionary
                        # uncomment this functionality if needed (could be slow due to file ops during scoring)
                        #sfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
                        #self.pickle.dump(dicts, sfile) 
                        #sfile.close()
                    else:    
                        df_add.at[index, cname] = cname_dict[cname_value]
                    
                    self.printlog ("JSON Loader: column " + cname + "; value: " + cname_value + " mapped")
                    
agent_{id} = cls_agent_{id}()
