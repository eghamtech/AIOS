#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv.bz2']
#key=tag_prefix;  type=constant;  value='CV'
#key=col_max_length;   type=constant;  value=200
#key=new_field_prefix; type=constant;  value=parsed_Actonomy_JSON_
#key=field_prefix_use_source_names;  type=constant;  value=True
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
# this agent creates new column from given field by converting XML into JSON
#
# source field is expected to be a dictionary field
#
# if "fields_source" parameter not specified then a field will be obtained randomly
# according to normal AIOS logic

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
    tag_prefix         = {tag_prefix}
    col_max_length     = {col_max_length}
    agent_name         = 'agent_' + str(result_id)

    field_prefix_use_source_names = {field_prefix_use_source_names}
    
    new_columns = []
    dict_cols   = []

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
        s = s.replace('?', '_qm_')
        s = s.replace('!', '_em_')

        s = s.replace('.', '_dot_')
        s = s.replace(',', '_coma_')
        s = s.replace(':', '_cln_')
        s = s.replace(';', '_scln_')

        s = re.sub('[^0-9a-zA-Z]+', '_', s)
        return  s
        
    def replace_xml_tags(xml_tag):
        xml_tag = str(xml_tag).replace('{http://actonomy.com/hrxml/2.5}', '')
        xml_tag = xml_tag.replace('{http://schemas.xmlsoap.org/soap/envelope/}', '')
        xml_tag = xml_tag.replace('{http://xmp.actonomy.com}', '')
        
        xml_tag = xml_tag.replace('StructuredXMLResume', '')
        xml_tag = xml_tag.replace('ContactInfo', 'CI')
        xml_tag = xml_tag.replace('ContactMethod', 'CM')
        xml_tag = xml_tag.replace('EmploymentHistory', 'EH')
        xml_tag = xml_tag.replace('EmployerOrg', 'EO')
        xml_tag = xml_tag.replace('EducationHistory', 'EDH')
        xml_tag = xml_tag.replace('SchoolOrInstitution', 'SOI')
        xml_tag = xml_tag.replace('Qualifications', 'QLS')
        xml_tag = xml_tag.replace('UserArea', '')
        xml_tag = xml_tag.replace('Classifications', 'CLS')
        xml_tag = xml_tag.replace('Value', '')
        xml_tag = xml_tag.replace('Competency', 'Skills')
        xml_tag = xml_tag.replace('LocationSummary', 'Location')
        xml_tag = xml_tag.replace('PositionHistory', 'Position')
        xml_tag = xml_tag.replace('AnyDate', 'YearMonth')

        return xml_tag

    def xml_actonomy_2json(self, xml_str, tag_prefix='hrx'):
        jout = {}
        tags = {}
        tc   = 0
        
        # missing tags to be filled with 'Unknown' in groups specified as keys in this dictionary
        mtags = {
            'EO'         : ['EOName'],
            'Position'   : ['Title'],
            'Location'   : ['Municipality', 'CountryCode']
        }
        
        def recursive_parse(children, tag_prefix, tc):

            tc = tc+1
            tags[tc] = []
            
            for child in list(children):

                child_text = str(child.text).replace("\n",' ').replace("\r",' ').strip()           
                
                child_tag_clean = self.replace_xml_tags(child.tag)          
                tags[tc].append(child_tag_clean)

                if self.replace_xml_tags(children.tag) == 'Position':
                    child_count = tags[tc].count(child_tag_clean)
                    if child_count > 1:
                        child_tag_clean = child_tag_clean + "_alv" + str(child_count)
                
                
                if tag_prefix == '':
                    child_tag = child_tag_clean
                else:
                    child_tag = tag_prefix + "_" + child_tag_clean
                    
                    if child.attrib.get('type') is not None:
                        child_tag = child_tag + '_' + child.attrib.get('type', '')

                if child.attrib.get('name') is not None:
                    if child_text == '' or child_text == 'None':
                        child_text = child.attrib['name']
                    else:
                        child_text = child.attrib['name'] + " | " + child_text

                if child_text != '' and child_text != 'None':
                    if jout.get(child_tag) is None:
                        jout[child_tag] = []

                    jout[child_tag].append(child_text)

                    # extract "weight" attribute and add it as separate json item
                    if child.attrib.get('weight') is not None:
                        w_tag = child_tag + '_' + clean_text(child_text)

                        if jout.get(w_tag) is None:
                            jout[w_tag] = []

                        jout[w_tag].append(float(child.attrib.get('weight')))                    

                # apply same function recursively for all children
                if len(list(child)) > 0:                    
                    recursive_parse(child, child_tag, tc)   
                else:               
                    # if no children exist for StartDate or EndDate tag then add empty YearMonth tag to make sure it is recorded
                    if child_tag_clean == 'EndDate' or child_tag_clean == 'StartDate' or child_tag_clean == 'DegreeDate':
                        child_tag_ym = child_tag + '_YearMonth'
                        if jout.get(child_tag_ym) is None:
                            jout[child_tag_ym] = []
                        jout[child_tag_ym].append('1600-01')
                        
            # add default value for missing children tags in specified groups        
            for tag in mtags:
                if self.replace_xml_tags(children.tag) == tag:
                    for ctag in mtags[tag]:
                        if ctag not in tags[tc]:
                            child_tag = tag_prefix + "_" + ctag
                            if jout.get(child_tag) is None:
                                jout[child_tag] = []

                            jout[child_tag].append('Unknown')
                        
        # - end of recursive function
        
        try:         
            root = etree_lxml.fromstring(xml_str.encode('utf-8'))
            recursive_parse(root, tag_prefix, tc)
        except Exception as e:
            print (e)
            print (str(datetime.now()), 'Error in XML')
            return {}

        return jout


    def run_on(self, df_run, apply_fun=False):
        self.dict_cols = []
        for i in range(0,len(self.data_defs)):
            col_name = self.data_defs[i].split("|")[0]
            self.dict_cols.append(col_name)

        self.new_columns = []
        new_col_name = self.new_field_prefix + '_' + str(self.result_id)
        self.new_columns.append(new_col_name)

        df_results     = []
        block_progress = 0
        total = len(df_run)
        block = int(total/50)
        
        for index, row in df_run.iterrows():
            row_str = ''
            for col_name in self.dict_cols:
                row_str += ' ' + str(row['dict_'+col_name])   # concatenate columns into one string
            row_str = row_str[1:]                             # remove space added during columns concatenation
            
            if row_str == '' or row_str == 'nan' or row_str == 'NaN' or row_str == 'None' or row_str.replace("\n",' ').replace("\r",' ').strip() == '':
                xml_parsed = json.dumps({})
            else:
                xml_parsed = json.dumps(self.xml_actonomy_2json(row_str, self.tag_prefix))

            df_results.append(xml_parsed) 

            block_progress += 1
            if (block_progress >= block):
                block_progress = 0
                print (str(datetime.now()), " rows processed: ", round((index+1)/total*100,0), "%")

        df_run[self.new_columns] = pd.DataFrame(df_results)   


    def run(self, mode):
        print ("enter run mode " + str(mode))

        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if i==0:
                self.df = pd.read_csv(workdir+file_name)[[col_name]]
            else:
                self.df = self.df.merge(pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)

            if os.path.isfile(workdir + 'dict_' + file_name):
                # load dictionary if it exists
                dict_temp = pd.read_csv(workdir + 'dict_' + file_name, dtype={'value': object}).set_index('key')["value"].to_dict()
                # replace column with its mapped value from dictionary
                self.df['dict_'+col_name] = self.df[col_name].map(dict_temp)

        self.run_on(self.df)
        nrow = len(self.df)

        # save and register each new column
        # all columns are dictionary fields
        for i in range(0,len(self.new_columns)):
            fld   = self.new_columns[i]
            fname = fld + self.out_file_extension

            fld_dict     = self.make_dict(self.df[fld].fillna(''))         # create dictionary of given text column  
            self.df[fld] = self.df[fld].fillna('').map(fld_dict)           # replace column values with corresponding values from dictionary

            # save dictionary for each text column into separate file
            pd.DataFrame(list(fld_dict.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+fname, encoding='utf-8')

            # save column of indexes
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",Y,"+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add, apply_fun=True)

agent_{id} = cls_agent_{id}()
