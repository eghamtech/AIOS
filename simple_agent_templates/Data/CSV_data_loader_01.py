#start_of_parameters
#key=source_filename;  type=constant;  value=enter_source_filename
#key=target;  type=constant;  value=enter_target
#end_of_parameters

# This script will scan your CSV file for string columns, convert them to dictionaries
# and create columns in AIOS Memory with data for each column in the 'source_filename' CSV file.
# Provide correct 'source_filename' in the agent parameters.
# Variables 'workdir' and 'trainfile' (target filename) must be setup in 'Constants' area of AIOS
# Parameter 'target' specifies column to be marked as the prediction target.

if 'dicts' not in globals():
    dicts = {}  # dict of dicts. each of dicts has structure: key=string, value=number

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    
    source_filename = "{source_filename}"
    target = "{target}"
    newfilename = trainfile
    
    def __init__(self):
        global dicts
        self.df = self.pd.read_csv(workdir+self.source_filename, encoding='utf8')
        print (self.df)
        self.char_cols = list(self.df.select_dtypes(include=['object']).columns)
        print ("source data loaded")
        print ("char columns:", self.char_cols)
        #self.dicts = {}
        for cname in self.char_cols:
            dicts[cname] = self.make_dict(self.df[cname].fillna(''))

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))

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
            if cname==self.target:
                is_target="Y"
            else:
                is_target="N"
            print ("#add_field:"+cname+","+is_dict+","+self.newfilename+","+is_target+","+str(nrow))
    
    def apply(self, df_add):
        global dicts
        for index, row in df_add.iterrows():
            for cname in df_add.columns:
                if cname in self.char_cols:
                    if not (row[cname] in dicts[cname]):
                        dicts[cname][row[cname]] = 1+max(dicts[cname].values())
                    df_add.at[index, cname] = dicts[cname][row[cname]]
                else:
                    df_add.at[index, cname] = row[cname]

agent_{id} = cls_agent_{id}()
