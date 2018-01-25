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
    
    source_filename = "{source_filename_json}"
    # target must be set on constants page
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
            str = cols[i]
            for ch in [".", ",", " ", "/", "(", ")", "?", "!"]:
                str = str.replace(ch, "_")
            new_cols.append(str)
            self.colmap[cols[i]] = str
        self.df.columns = new_cols
        
        print ("processing DATETIME columns...")
        self.date_cols = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            if item["data_type"]=='DATETIME':
                self.date_cols.append(item["heading"])
                print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                self.df[self.colmap[item["heading"]]+'_Y'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None else 0)
                self.df[self.colmap[item["heading"]]+'_M'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None else 0)
                self.df[self.colmap[item["heading"]]+'_D'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None else 0)
                self.df[self.colmap[item["heading"]]+'_WD'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None else 0)
                self.df[self.colmap[item["heading"]]+'_TS'] = self.df[self.colmap[item["heading"]]].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None else 0)
                self.df = self.df.drop(self.colmap[item["heading"]], 1)
        
        print ("processing FREETEXT/LARGETEXT columns")
        self.char_cols = [] #list(self.df.select_dtypes(include=['object']).columns)
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            if item["data_type"]=='FREETEXT' or item["data_type"]=='LARGETEXT':
                print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                self.char_cols.append(self.colmap[item["heading"]])
                
        print ("char columns:", self.char_cols)
        
        print ("processing OUTCOME columns")
        self.target_cols = []
        for i in range(0, len(json_data["model_definition"]["layout"]["columns"])):
            item = json_data["model_definition"]["layout"]["columns"][i]
            if item["analysis"]=='outcome':
                print (i, item["analysis"], "---------", item["heading"], "---------", item["data_type"])
                self.target_cols.append(self.colmap[item["heading"]])
                
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
            self.pd.DataFrame(list(dict1.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+cname+'.csv', encoding='utf8')    #save new column dict
        
        self.df.to_csv(workdir+self.newfilename, index=False)
        
        nrow = len(self.df)

        for cname in self.df.columns:
            if cname in self.char_cols:
                is_dict="Y"
            else:
                is_dict="N"
            if cname in self.target_cols:
                is_target="Y"
            else:
                is_target="N"
            print ("#add_field:"+cname+","+is_dict+","+self.newfilename+","+is_target+","+str(nrow))
    
    def apply(self, df_add):
        global dicts
        for cname in self.date_cols:
            df_add[cname+'_Y'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).year if x!=None else 0)
            df_add[cname+'_M'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).month if x!=None else 0)
            df_add[cname+'_D'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).day if x!=None else 0)
            df_add[cname+'_WD'] = df_add[cname].apply(lambda x: self.dateutil.parser.parse(x).weekday() if x!=None else 0)
            df_add[cname+'_TS'] = df_add[cname].apply(lambda x: self.calendar.timegm(self.dateutil.parser.parse(x).timetuple()) if x!=None else 0)
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
