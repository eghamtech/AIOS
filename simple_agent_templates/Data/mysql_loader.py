#start_of_parameters
#key=server_address;  type=constant;  value=enter_server_address
#key=ssh_port;  type=constant;  value=enter_ssh_port
#key=ssh_username;  type=constant;  value=enter_ssh_username
#key=ssh_password;  type=constant;  value=enter_ssh_password
#key=mysql_user;  type=constant;  value=enter_mysql_user
#key=mysql_password;  type=constant;  value=enter_mysql_password
#key=mysql_database;  type=constant;  value=enter_mysql_database
#key=sql_to_load_data;  type=constant;  value=enter_sql_to_load_data
#key=date_column;  type=constant;  value=enter_date_column
#key=char_columns;  type=constant;  value=enter_char_columns
#key=apply_log_columns;  type=constant;  value=enter_apply_log_columns
#key=final_list_of_columns;  type=constant;  value=enter_final_list_of_columns
#end_of_parameters

if 'dicts' not in globals():
    dicts = {}
    
class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import datetime
    import pandas as pd
    import numpy as np
    from sshtunnel import SSHTunnelForwarder
    import pymysql
    
    date_column = "{date_column}"
    char_cols = "{char_columns}".split(",")
    apply_log_cols = "{apply_log_columns}".split(",")
    newfilename = trainfile
    targetcol = target

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))
    
    def run(self, mode):
        global dicts
        print ("enter run mode " + str(mode))
        with self.SSHTunnelForwarder(
              ("{server_address}", {ssh_port}),
              ssh_username="{ssh_username}",
              ssh_password="{ssh_password}",
              remote_bind_address=("127.0.0.1", 3306)
         ) as server:
            print ("connect mysql...")
            myConnection = self.pymysql.connect( host="127.0.0.1", port=server.local_bind_port, user="{mysql_user}", passwd="{mysql_password}", db="{mysql_database}" )
            print ("start load mysql...")
            df = self.pd.read_sql('{sql_to_load_data}', con=myConnection)
            print ("close mysql connection...")
            myConnection.close()

        if len(self.date_column)>0:
            print ("processing date column year...")
            df[self.date_column + "_year"] = df[self.date_column].apply(lambda x: self.datetime.datetime.strptime(x, "%Y-%m-%d 00:00").year)
            print ("processing date column month...")
            df[self.date_column + "_month"] = df[self.date_column].apply(lambda x: self.datetime.datetime.strptime(x, "%Y-%m-%d 00:00").month)
            print ("processing date column day...")
            df[self.date_column + "_day"] = df[self.date_column].apply(lambda x: self.datetime.datetime.strptime(x, "%Y-%m-%d 00:00").day)
            print ("drop date column...")
            df.drop(self.date_column, axis=1, inplace=True)

        print("saving dicts...")
        for f in self.char_cols:
            print (f)
            dicts[f] = self.make_dict(df[f].fillna(''))
            df[f] = df[f].map(dicts[f])
            self.pd.DataFrame(list(dicts[f].items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+f+'.csv', encoding='utf-8')    #save new column dict
        
        print("convert column(s) to log...")
        for f in self.apply_log_cols:
            df[f+"log"] = self.np.log(df[f].astype(float))
            df.drop(f, axis=1, inplace=True)
        
        print ("save dataframe on local disk...")
        df.to_csv(workdir+self.newfilename, index=False)
        
        nrow = len(df)

        for cname in df.columns:
            if cname in self.char_cols:
                is_dict="Y"
            else:
                is_dict="N"
            if cname==self.targetcol:
                is_target="Y"
            else:
                is_target="N"
            print ("#add_field:"+cname+","+is_dict+","+self.newfilename+","+is_target+","+str(nrow))

    def apply(self, df_add):
        global dicts
        for f in "{final_list_of_cols}":
            if f not in df_add.columns:
                df_add[f] = float('nan')
        for f in df_add.columns:
            if f in self.char_cols:
                if f not in dicts:
                    dicts[f] = self.pd.read_csv(workdir+'dict_'+f+'.csv', dtype={'value': object}).set_index('value')["key"].to_dict()
                df_add[f] = df_add[f].map(dicts[f])

agent_{id} = cls_agent_{id}()
