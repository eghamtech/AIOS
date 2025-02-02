#start_of_parameters
#key=server_address;  type=constant;  value=enter_server_address
#key=ssh_port;  type=constant;  value=enter_ssh_port
#key=ssh_username;  type=constant;  value=enter_ssh_username
#key=ssh_password;  type=constant;  value=enter_ssh_password;  is_password=1
#key=mysql_user;  type=constant;  value=enter_mysql_user
#key=mysql_password;  type=constant;  value=enter_mysql_password;  is_password=1
#key=mysql_database;  type=constant;  value=enter_mysql_database
#key=data_ids;  type=constant;  value=enter_data_ids
#end_of_parameters

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import datetime
    import pandas as pd
    import numpy as np
    from sshtunnel import SSHTunnelForwarder
    import pymysql
    
    newfilename = trainfile
    targetcol = target
    data_ids = "{data_ids}"
    df_out = None

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
            
            a_data = self.data_ids.split(",")
            for data_id in reversed(a_data):
                print(data_id)
                
                df = self.pd.read_sql('select date, open, high, low, close, 0 as is_last_row, 0 as target from tblData_' + data_id + ' order by date desc', con=myConnection)
                df.at[0, "is_last_row"] = 1
                print(df.head())

                if self.df_out is None:
                    self.df_out = df.copy()
                else:
                    self.df_out = self.pd.concat([self.df_out, df]).reset_index(drop=True)
            
            print ("close mysql connection...")
            myConnection.close()

        print ("save dataframe on local disk...")
        self.df_out.to_csv(workdir+self.newfilename, index=False)
        
        nrow = len(self.df_out)

        for cname in df.columns:
            is_dict="N"
            is_target="Y" if cname=="target" else "N"
            print ("#add_field:"+cname+","+is_dict+","+self.newfilename+","+is_target+","+str(nrow))

    def apply(self, df_add):
        return

agent_{id} = cls_agent_{id}()
