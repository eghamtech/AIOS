#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv']
#key=fields_source_file_or_text;  type=constant;  value=False
#key=xml_template; type=constant;  value=candidate_actonomy | candidate_daxtra_native | vacancy_actonomy | vacancy_daxtra_native
#key=replace_bbtags;  type=constant;  value=True
#key=col_max_length;   type=constant;  value=200
#key=new_field_prefix; type=constant;  value=parsed_Actonomy_
#key=field_prefix_use_source_names;  type=constant;  value=True
#key=proxy_http;  type=constant;  value=http://127.0.0.1
#key=proxy_https; type=constant;  value=http://127.0.0.1
#key=actonomy_url;   type=constant;  value=https://127.0.0.1
#key=actonomy_user;  type=constant;  value=username
#key=actonomy_pass;  type=constant;  value=password;  is_password=1
#key=include_columns_type;  type=constant;  value=is_dict_only
#key=include_columns_containing; type=constant;  value=
#key=ignore_columns_containing;  type=constant;  value='%ev_field%' and '%onehe_%'
#key=out_file_extension;  type=constant;  value=.csv.bz2
#end_of_parameters

# AICHOO OS Simple Agent
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new column from single dictionary field by
# sending the text to Actonomy server for parsing as CV or Vacancy
#
# in this version of agent the source field must be a dictionary field
#
# parameter xml_template specifies whether to use Daxtra parser only or Actonomy on top of Daxtra
# and whether to parse as CV or Vacancy
#
# if "fields_source" parameter not specified then a field will be obtained randomly
# according to normal AIOS logic (uncomment line 170)

import warnings
warnings.filterwarnings("ignore")
import gc
gc.collect()

import pandas as pd
import os.path, bz2, pickle, re, json, base64
import requests
import xml.etree.ElementTree as ET

from lxml import etree as etree_lxml
from datetime import datetime

class cls_agent_{id}:
    data_defs          = {fields_source}
    # obtain a unique ID for the current instance
    result_id          = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix   = "{new_field_prefix}"
    out_file_extension = "{out_file_extension}"
    replace_bbtags     = {replace_bbtags}
    col_max_length     = {col_max_length}
    agent_name         = 'agent_' + str(result_id)

    field_prefix_use_source_names = {field_prefix_use_source_names}
    fields_source_file_or_text    = {fields_source_file_or_text}     # if True then source fields contain path to file to load content from
    
    new_columns = []
    dict_cols   = []

    PROXIES = {
        "https" : "{proxy_https}",
        "http"  : "{proxy_http}"
    }

    auth_str       = "{actonomy_user}:{actonomy_pass}"
    actonomy_url   = "{actonomy_url}"
    
    body_candidate_daxtra_native = """<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xmp="http://xmp.actonomy.com">
            <soapenv:Header/>
            <soapenv:Body>
                <xmp:parse>
                    <action>
                        <provider>daxtra</provider>
                        <documents>
                        <content>{b64content}
                        </content>
                        <name>candidate_daxtra_native.pdf</name>
                        </documents>
                        <documentType>candidate</documentType>
                        <resultFormat>native</resultFormat>
                    </action>
                </xmp:parse>
            </soapenv:Body>
            </soapenv:Envelope>"""

    body_vacancy_daxtra_native = """<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xmp="http://xmp.actonomy.com">
            <soapenv:Header/>
            <soapenv:Body>
                <xmp:parse>
                    <action>
                        <provider>daxtra</provider>
                        <documents>
                        <content>{b64content}
                        </content>
                        <name>vacancy_daxtra_native.pdf</name>
                        </documents>
                        <documentType>job</documentType>
                        <resultFormat>native</resultFormat>
                    </action>
                </xmp:parse>
            </soapenv:Body>
            </soapenv:Envelope>"""

    body_candidate_actonomy = """<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xmp="http://xmp.actonomy.com">
            <soapenv:Header/>
            <soapenv:Body>
                <xmp:parse>
                    <action>
                        <provider>daxtra</provider>
                        <documents>
                        <content>{b64content}
                        </content>
                        <name>candidate_actonomy.pdf</name>
                        </documents>
                        <documentType>candidate</documentType>
                        <resultFormat>hrxml-actonomy</resultFormat>
                    </action>
                </xmp:parse>
            </soapenv:Body>
            </soapenv:Envelope>"""

    body_vacancy_actonomy = """<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xmp="http://xmp.actonomy.com">
            <soapenv:Header/>
            <soapenv:Body>
                <xmp:parse>
                    <action>
                        <provider>daxtra</provider>
                        <documents>
                        <content>{b64content}
                        </content>
                        <name>vacancy_actonomy.pdf</name>
                        </documents>
                        <documentType>job</documentType>
                        <resultFormat>hrxml-actonomy</resultFormat>
                    </action>
                </xmp:parse>
            </soapenv:Body>
            </soapenv:Envelope>"""

    xml_body_templates = {
        "candidate_daxtra_native" : body_candidate_daxtra_native,
        "candidate_actonomy"      : body_candidate_actonomy,
        "vacancy_daxtra_native"   : body_vacancy_daxtra_native,
        "vacancy_actonomy"        : body_vacancy_actonomy
    }
    xml_template = xml_body_templates["{xml_template}"]

    def is_set(self, s):
        try:
            not_empty = (len(s)>0 and s!="0")
        except:
            not_empty = True
        return not_empty

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))

    def __init__(self):
        #if not self.is_set(self.data_defs):
        #    self.data_defs = ["{_random_dict_distinct}"]

        if self.field_prefix_use_source_names:                   
            # concatenate all source column names into new field prefix
            col_max_length = int(200 / len(self.data_defs))             # allow 200 characters max combined col name length
            for i in range(0,len(self.data_defs)):
                col_name = self.data_defs[i].split("|")[0]
                col_name = col_name[:col_max_length]                   # only take first col_max_length chars from each column
                self.new_field_prefix = self.new_field_prefix + '_' + col_name

    
    def send_to_actonomy(self, file, body, row_index, file_or_text=True):

        if file_or_text:
            # file parameter is file to be loaded
            with open(file, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
        else:
            # file parameter is a string already
            encoded_string = base64.b64encode(file.encode())

        # replace content keyword in XML template with actual content encoded as base64
        body_subm = body.format(b64content=encoded_string.decode('utf-8'))

        # prepare authentication    
        encoded = base64.b64encode(self.auth_str.encode('ascii'))
        encoded = str(encoded)[2:-1]
        auth    = "Basic " + encoded
        headers = {'content-type': 'text/soap+xml', 'authorization': auth} 

        attempts       = 0
        not_successful = True

        while not_successful and attempts < 5:
            try:
                r = requests.post(self.actonomy_url, proxies=self.PROXIES, data=body_subm, headers=headers, verify=False)
            
                if r.status_code != 200 and r.status_code != 404 and r.status_code != 415:
                    r.raise_for_status()

                not_successful = False
            except requests.exceptions.RequestException as e:
                attempts += 1
                print (e)
                if attempts < 5:
                    print (str(datetime.now()), 'Error API request at dict key: ', row_index, '; retry attempt: ', attempts)
                else:
                    print (str(datetime.now()), 'Error API request at dict key: ', row_index, '; FATAL no more attempts')
                    return False  
        
        try:
            root = ET.fromstring(r.text)

            content = root.find(
                '{http://schemas.xmlsoap.org/soap/envelope/}Body'
            ).find(
                '{http://xmp.actonomy.com}parseResponse'
            ).find('return').find(
                'documents'
            ).find('content').text

            base64_message = base64.b64decode(content).decode('utf-8')
        except Exception as e:
            print (e)
            print (str(datetime.now()), 'Error in Actonomy returned XML at dict key: ', row_index)
            return False
        
        return base64_message


    def run_on(self, df_run, col_dict={}, apply_fun=False):
        col_name = self.data_defs[0].split("|")[0]

        self.new_columns = []
        new_col_name = self.new_field_prefix + '_' + str(self.result_id)
        self.new_columns.append(new_col_name)

        col_dict_new   = {}

        if apply_fun:
            fld_dict = self.make_dict(df_run['dict'+col_name].fillna(''))             # create dictionary of given text column  
            df_run[new_col_name] = df_run['dict'+col_name].fillna('').map(fld_dict)   # replace column values with corresponding values from dictionary
            col_dict = {v:k for k,v in fld_dict.items()}                              # reverse new dictionary so it can be iterated over keys
        else:
            df_run[new_col_name] = df_run[col_name]    # keys are the same as original column

        block_progress = 0
        index = 0
        total = len(col_dict)
        block = int(total/50)
        
        for k,v in col_dict.items():
            row_str = str(v)

            if self.replace_bbtags:
                row_str = row_str.replace('[','<').replace(']','>').replace(u'\xa0', u' ')
            
            if row_str == '' or row_str == 'nan' or row_str == 'NaN' or row_str == 'None' or row_str.replace("\n",' ').replace("\r",' ').strip() == '':
                xml_parsed = ''
            else:
                xml_parsed = self.send_to_actonomy(row_str, self.xml_template, k, file_or_text=self.fields_source_file_or_text)

                if xml_parsed == False:
                    xml_parsed = ''

            col_dict_new[k] = xml_parsed

            block_progress += 1
            index += 1
            if (block_progress >= block):
                block_progress = 0
                print (str(datetime.now()), " keys processed: ", round((index)/total*100,0), "%")
        
        df_run['dict_' + new_col_name] = df_run[new_col_name].map(col_dict_new)

        return col_dict_new


    def run(self, mode):
        print ("enter run mode " + str(mode))

        col_name  = self.data_defs[0].split("|")[0]
        file_name = self.data_defs[0].split("|")[1]

        self.df = pd.read_csv(workdir+file_name)[[col_name]]

        if os.path.isfile(workdir + 'dict_' + file_name):
            # load dictionary if it exists
            dict_temp = pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
            # replace column with its mapped value from dictionary
            self.df['dict_'+col_name] = self.df[col_name].map(dict_temp)
        else:
            # no dictionary file found
            raise ValueError('Aborting: No dictionary file found for the field: ', col_name)
            return

        fld_dict = self.run_on(self.df, dict_temp)
        nrow = len(self.df)

        # save and register new column with new dictionary
        fld   = self.new_columns[0]
        fname = fld + self.out_file_extension

        # save dictionary into separate file
        pd.DataFrame(list(fld_dict.items()), columns=['key','value']).to_csv(workdir+'dict_'+fname, encoding='utf-8')

        # save column of indexes
        self.df[[fld]].to_csv(workdir+fname)
        print ("#add_field:"+fld+",Y,"+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add, apply_fun=True)

agent_{id} = cls_agent_{id}()
