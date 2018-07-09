#start_of_parameters
#key=source_filename_json;  type=constant;  value=enter_source_filename_json
#key=unique_id_column;  type=constant;  value=enter_unique_id_column
#end_of_parameters

# Processes JSON file which has "training_data" and "model_definition" objects according to below specification in Wiki:
# https://github.com/eghamtech/AIOS/wiki/Input-data-JSON-format-01

#multiple files support assumes that files named like this:
#input_data.json -- main file, mandatory
#input_data_1.json -- optional
#input_data_2.json -- optional
#etc...

if 'dicts' not in globals():
    dicts = {}
    
class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import json, datetime, calendar, dateutil.parser
    import re, bz2, pickle, os.path
    
    source_filename = "{source_filename_json}"
    unique_id_column = "{unique_id_column}"
    # newfilename = trainfile
    # obtain a unique ID for the current instance
    result_id = {id}
    agent_name = 'agent_' + str(result_id)
    colmap = {}
    char_cols = []
    date_cols = []
    lookup_cols = []

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def __init__(self):
        global dicts
        
        # if saved model for dictionaries already exists then load it from filesystem
        if self.os.path.isfile(workdir + self.agent_name + '.model'):
            rfile = self.bz2.BZ2File(workdir + self.agent_name + '.model', 'r')
            dicts = self.pickle.load(rfile)
            rfile.close()
            
            self.colmap = dicts[self.agent_name + '.colmap']
            self.char_cols = dicts[self.agent_name + '.char_cols']
            self.date_cols = dicts[self.agent_name + '.date_cols']
            self.lookup_cols = dicts[self.agent_name + '.lookup_cols']
  
         
    def run(self, mode):
        from datetime import datetime
        
        global dicts
        print ("enter run mode " + str(mode))
        
        print (str(datetime.now()), " loading json file to dataframe...")
        
        jsons = []
        with open(workdir + self.source_filename, encoding='utf-8') as f1:
            json_data = self.json.load(f1)
            f1.close()
        jsons = [json_data]
        
        filename, file_extension = self.os.path.splitext(workdir + self.source_filename)
        inext = 1
        fname = filename + "_" + str(inext) + file_extension
        while self.os.path.isfile(fname):
            print ("found next file: " + fname)
            with open(fname, encoding='utf-8') as f1:
                json_data = self.json.load(f1)
                f1.close()
            jsons.append(json_data)
            inext += 1
            fname = filename + "_" + str(inext) + file_extension
        
        print (str(datetime.now()), " creating dataframe...")       
        self.colmap = {}
        for idata in range(0, len(jsons)):
            json_data = jsons[idata]
            df_new = self.pd.DataFrame().from_dict(json_data["training_data"])
            new_cols = []
            for c in df_new.columns:
                str1 = c
                str1 = self.re.sub('[^0-9a-zA-Z]+', '_', str1)
                str1 = str1 + "_" + str(self.result_id)
                new_cols.append(str1)
                self.colmap[c] = str1
            df_new.columns = new_cols
            
            if idata==0:
                self.df = df_new
            else:
                self.df = self.df.append(df_new).reset_index(drop=True)
        
        if len(self.unique_id_column) > 0:
            print ("use column ", self.unique_id_column, " as unique id")
            self.df = self.df.groupby(self.unique_id_column, as_index=False).last().reset_index(drop=True)
          
        print (str(datetime.now()), " processing DATETIME columns...")
        self.date_cols = []
        for json_data in jsons:
            print("taking json")
            for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
                item = json_data["model_definition"]["layout"]["columns"][i]               # get column definition item
                cname = self.colmap[item["heading"]]                                       # obtain new name for the given column

                if item["data_type"]=='DATETIME':      
                    if cname not in self.date_cols:
                        self.date_cols.append(cname)
                        print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                        self.df[cname+'_Y'] = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
                        self.df[cname+'_M'] = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
                        self.df[cname+'_D'] = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
                        self.df[cname+'_WD'] = self.df[cname].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
                        self.df[cname+'_TS'] = self.df[cname].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
                        self.df = self.df.drop(cname, 1)
        
        print (str(datetime.now()), " processing FREETEXT/LARGETEXT columns")
        self.char_cols = []
        for json_data in jsons:
            print("taking json")
            for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
                item = json_data["model_definition"]["layout"]["columns"][i]               # get column definition item
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
        for json_data in jsons:
            print("taking json")
            for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
                item = json_data["model_definition"]["layout"]["columns"][i]               # get column definition item
                cname = self.colmap[item["heading"]]                                       # obtain new name for the given column

                if item["data_type"]=='LOOKUP':
                    if cname not in self.lookup_cols:
                        print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"], "---", item["meta"]["lookup_id"], "---", item["meta"]["lookup_type"])
                        self.lookup_cols.append(cname)

                        dict_lookup = {}
                        lookup_id = str(item["meta"]["lookup_id"])
                        lookup_type = item["meta"]["lookup_type"]
                        for key in json_data["static_data"]["lookups"][lookup_type][lookup_id]["lookup_values"].keys():
                            dict_lookup[key] = json_data["static_data"]["lookups"][lookup_type][lookup_id]["lookup_values"][key]["value"]

                        dicts[cname] = dict_lookup
                        self.pd.DataFrame(list(dict_lookup.items()), columns=['key', 'value']).to_csv(workdir+'dict_'+self.colmap[item["heading"]]+'.csv', encoding='utf-8')    #save new column dict
        # print ("lookup columns:", self.lookup_cols)
         
        print (str(datetime.now()), " processing DATA/OUTCOME columns")
        self.target_cols = []
        self.use_for_models = []
        for json_data in jsons:
            print("taking json")
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
        dicts[self.agent_name + '.colmap'] = self.colmap
        dicts[self.agent_name + '.char_cols'] = self.char_cols
        dicts[self.agent_name + '.date_cols'] = self.date_cols
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
            output_column = cname
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
                df_add[cshort] = float('nan')
            else:
                df_add.rename(index=str, columns={creal: cshort}, inplace=True)
                           
        for cname in self.date_cols:
            df_add[cname+'_Y'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_M'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_D'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_WD'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_TS'] = df_add[cname].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
            df_add.drop(cname, axis=1, inplace=True)
            
        for index, row in df_add.iterrows():                                      # iterate over each row in df_add 
            for cname in df_add.columns:
                if (cname in self.char_cols) or (cname in self.lookup_cols):
                    # df_add[cname] = df_add[cname].map(dicts[cname])
                    if not (row[cname] in dicts[cname]):                          # if value in current row and column not in dictionary
                        dicts[cname][row[cname]] = 1+max(dicts[cname].values())   # create new key in dictionary with max+1 value
                    df_add.at[index, cname] = dicts[cname][row[cname]]

agent_{id} = cls_agent_{id}()
