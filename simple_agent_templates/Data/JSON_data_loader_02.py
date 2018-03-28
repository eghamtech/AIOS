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
    import json
    import datetime
    import calendar
    import dateutil.parser
    import re
    
    source_filename = "{source_filename_json}"
    unique_id_column = "{unique_id_column}"
    newfilename = trainfile
    colmap = {}
    
    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def __init__(self):
        import os.path
        global dicts
        
        print ("loading json file to dataframe...")
        
        jsons = []
        
        with open(workdir + self.source_filename, encoding='utf-8') as f1:
            json_data = self.json.load(f1)
        jsons = [json_data]
        
        filename, file_extension = os.path.splitext(workdir + self.source_filename)
        inext = 1
        fname = filename + "_" + str(inext) + file_extension
        while os.path.isfile(fname):
            print ("found next file: " + fname)
            with open(fname, encoding='utf-8') as f1:
                json_data = self.json.load(f1)
            jsons.append(json_data)
            inext += 1
            fname = filename + "_" + str(inext) + file_extension
        
        
        print ("creating dataframe...")
        
        for idata in range(0, len(jsons)):
            json_data = jsons[idata]
            df_new = self.pd.DataFrame().from_dict(json_data["training_data"])
            new_cols = []
            for c in df_new.columns:
                str1 = c
                str1 = self.re.sub('[^0-9a-zA-Z]+', '_', str1)
                new_cols.append(str1)
                self.colmap[c] = str1
            df_new.columns = new_cols
            
            if idata==0:
                self.df = df_new
            else:
                self.df = self.df.append(df_new).reset_index(drop=True)
        
        if len(self.unique_id_column) > 0:
            print ("use column", self.unique_id_column, "as unique id")
            self.df = self.df.groupby(self.unique_id_column, as_index=False).last().reset_index(drop=True)
            
        print ("processing DATETIME columns...")
        self.date_cols = []
        for json_data in jsons:
            print("taking json")
            for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
                item = json_data["model_definition"]["layout"]["columns"][i]
                if item["data_type"]=='DATETIME':
                    if self.colmap[item["heading"]] not in self.date_cols:
                        self.date_cols.append(self.colmap[item["heading"]])
                        print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                        self.df[self.colmap[item["heading"]]+'_Y'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
                        self.df[self.colmap[item["heading"]]+'_M'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
                        self.df[self.colmap[item["heading"]]+'_D'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
                        self.df[self.colmap[item["heading"]]+'_WD'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
                        self.df[self.colmap[item["heading"]]+'_TS'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
                        self.df = self.df.drop(self.colmap[item["heading"]], 1)
        
        print ("processing FREETEXT/LARGETEXT columns")
        self.char_cols = [] #list(self.df.select_dtypes(include=['object']).columns)
        for json_data in jsons:
            print("taking json")
            for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
                item = json_data["model_definition"]["layout"]["columns"][i]
                if item["data_type"]=='FREETEXT' or item["data_type"]=='LARGETEXT':
                    if self.colmap[item["heading"]] not in self.char_cols:
                        print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                        self.char_cols.append(self.colmap[item["heading"]])
        print ("char columns:", self.char_cols)
        
        print ("processing LOOKUP columns")
        self.lookup_cols = []
        for json_data in jsons:
            print("taking json")
            for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
                item = json_data["model_definition"]["layout"]["columns"][i]
                if item["data_type"]=='LOOKUP':
                    if self.colmap[item["heading"]] not in self.lookup_cols:
                        print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"], "---", item["meta"]["lookup_id"], "---", item["meta"]["lookup_type"])
                        self.lookup_cols.append(self.colmap[item["heading"]])
                        dict_lookup = {}
                        lookup_id = str(item["meta"]["lookup_id"])
                        lookup_type = item["meta"]["lookup_type"]
                        for key in json_data["static_data"]["lookups"][lookup_type][lookup_id]["lookup_values"].keys():
                            dict_lookup[key] = json_data["static_data"]["lookups"][lookup_type][lookup_id]["lookup_values"][key]["value"]
                        self.pd.DataFrame(list(dict_lookup.items()), columns=['key', 'value']).to_csv(workdir+'dict_'+self.colmap[item["heading"]]+'.csv', encoding='utf-8')    #save new column dict
        print ("lookup columns:", self.lookup_cols)
        
        
        print ("processing OUTCOME columns")
        self.target_cols = []
        self.use_for_models = []
        for json_data in jsons:
            print("taking json")
            for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
                item = json_data["model_definition"]["layout"]["columns"][i]
                if item["analysis"]=='outcome':
                    if self.colmap[item["heading"]] not in self.target_cols:
                        print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                        self.target_cols.append(self.colmap[item["heading"]])
                        self.use_for_models.append(self.colmap[item["heading"]])
                        self.df[self.colmap[item["heading"]]] = self.df[self.colmap[item["heading"]]].astype(int)
                elif item["analysis"]=='data':
                    if self.colmap[item["heading"]] not in self.use_for_models:
                        self.use_for_models.append(self.colmap[item["heading"]])
                        if self.colmap[item["heading"]] in self.date_cols:
                            self.use_for_models.append(self.colmap[item["heading"]]+'_Y')
                            self.use_for_models.append(self.colmap[item["heading"]]+'_M')
                            self.use_for_models.append(self.colmap[item["heading"]]+'_D')
                            self.use_for_models.append(self.colmap[item["heading"]]+'_WD')
                            self.use_for_models.append(self.colmap[item["heading"]]+'_TS')
                
        print ("target columns:", self.target_cols)
        
        from numpy import nan
        self.df.fillna(value=nan, inplace=True)
        
        print ("making dicts...")
        for cname in self.char_cols:
            dicts[cname] = self.make_dict(self.df[cname].fillna(''))
            
        print ("done")
        
    def run(self, mode):
        global dicts
        print ("enter run mode " + str(mode))
        for cname in self.char_cols:
            dict1 = dicts[cname]
            self.df[cname] = self.df[cname].fillna('').map(dict1)
            self.pd.DataFrame(list(dict1.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+cname+'.csv', encoding='utf-8')    #save new column dict
        
        self.df.to_csv(workdir+self.newfilename, index=False)
        
        nrow = len(self.df)

        for cname in self.df.columns:
            if (cname in self.char_cols) or (cname in self.lookup_cols):
                is_dict="Y"
            else:
                is_dict="N"
            if cname in self.target_cols:
                is_target="Y"
            else:
                is_target="N"
            if cname in self.use_for_models:
                is_use_for_models="Y"
            else:
                is_use_for_models="N"
            print ("#add_field:"+cname+","+is_dict+","+self.newfilename+","+is_target+","+str(nrow)+","+is_use_for_models)
    
    def apply(self, df_add):
        global dicts
        for creal, cshort in self.colmap.items():
            if cshort not in df_add.columns:
                df_add[cshort] = float('nan')
        for cname in self.date_cols:
            df_add[cname+'_Y'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_M'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_D'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_WD'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_TS'] = df_add[cname].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
            df_add.drop(cname, axis=1, inplace=True)
        for cname in df_add.columns:
            if cname in self.char_cols:
                df_add[cname] = df_add[cname].map(dicts[cname])

agent_{id} = cls_agent_{id}()
