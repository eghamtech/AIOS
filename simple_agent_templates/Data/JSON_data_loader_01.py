#start_of_parameters
#key=source_filename_json;  type=constant;  value=enter_source_filename_json
#end_of_parameters

# Processes JSON file which has "training_data" and "model_definition" objects according to below specification in Wiki:
# https://github.com/eghamtech/AIOS/wiki/Input-data-JSON-format-01

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
    newfilename = trainfile
    colmap = {}
    
    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def __init__(self):
        global dicts
        
        print ("loading json file to dataframe...")
        
        with open(workdir + self.source_filename, encoding='utf-8') as f1:
            json_data = self.json.load(f1)
        
        print ("creating dataframe...")
        
        self.df = self.pd.DataFrame().from_dict(json_data["training_data"])
        cols = self.df.columns
        new_cols = []
        for i in range(0, len(cols)):
            str1 = cols[i]
            #for ch in [".", ",", " ", "/", "(", ")", "?", "!"]:
            #    str1 = str1.replace(ch, "_")
            str1 = self.re.sub('[^0-9a-zA-Z]+', '_', str1)
            new_cols.append(str1)
            self.colmap[cols[i]] = str1
        self.df.columns = new_cols
        
        print ("processing DATETIME columns...")
        self.date_cols = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            if item["data_type"]=='DATETIME':
                self.date_cols.append(item["heading"])
                print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                self.df[self.colmap[item["heading"]]+'_Y'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
                self.df[self.colmap[item["heading"]]+'_M'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
                self.df[self.colmap[item["heading"]]+'_D'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
                self.df[self.colmap[item["heading"]]+'_WD'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
                self.df[self.colmap[item["heading"]]+'_TS'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
                self.df = self.df.drop(self.colmap[item["heading"]], 1)
        
        print ("processing FREETEXT/LARGETEXT columns")
        self.char_cols = [] #list(self.df.select_dtypes(include=['object']).columns)
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            if item["data_type"]=='FREETEXT' or item["data_type"]=='LARGETEXT':
                print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                self.char_cols.append(self.colmap[item["heading"]])
        print ("char columns:", self.char_cols)
        
        print ("processing LOOKUP columns")
        self.lookup_cols = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            if item["data_type"]=='LOOKUP':
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
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            if item["analysis"]=='outcome':
                print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                self.target_cols.append(self.colmap[item["heading"]])
                self.use_for_models.append(self.colmap[item["heading"]])
                self.df[self.colmap[item["heading"]]] = self.df[self.colmap[item["heading"]]].astype(int)
            elif item["analysis"]=='data':
                self.use_for_models.append(self.colmap[item["heading"]])
                if item["heading"] in self.date_cols:
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
        for cname in self.date_cols:
            df_add[cname+'_Y'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_M'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_D'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_WD'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None and self.pd.notnull(x) else 0)
            df_add[cname+'_TS'] = df_add[cname].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None and self.pd.notnull(x) else 0)
            df_add = df_add.drop(cname, 1)
        for index, row in df_add.iterrows():
            for cname in df_add.columns:
                if cname in self.char_cols:
                    if not (row[cname] in dicts[cname]):
                        dicts[cname][row[cname]] = 1+max(dicts[cname].values())
                    df_add.at[index, cname] = dicts[cname][row[cname]]
                else:
                    df_add.at[index, cname] = row[cname]

agent_{id} = cls_agent_{id}()
