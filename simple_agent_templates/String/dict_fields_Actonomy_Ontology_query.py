#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv.bz2','dict_field2|dict_field2.csv.bz2']
#key=tag_prefix;  type=constant;  value='CV'
#key=col_max_length;   type=constant;  value=200
#key=new_field_prefix; type=constant;  value=parsed_Actonomy_JSON_
#key=field_prefix_use_source_names;  type=constant;  value=True
#key=proxy_http;  type=constant;  value=http://127.0.0.1
#key=proxy_https; type=constant;  value=http://127.0.0.1
#key=actonomy_url;   type=constant;  value=https://127.0.0.1/v5_7/OntologyService
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
# this agent creates new column from given field stored as rows of JSON objects
# by parsing specified keys through Actonomy Ontology
#
# source field must be a dictionary field
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

    PROXIES = {
        "https" : "{proxy_https}",
        "http"  : "{proxy_http}"
    }

    auth_str       = "{actonomy_user}:{actonomy_pass}"
    actonomy_url   = "{actonomy_url}"

    body_find_terms = """<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xmp="http://xmp.actonomy.com">
                            <soapenv:Body>
                                <xmp:findTerms>
                                    <action>
                                        <text>{body_text}</text>
                                        <searchAlgorithm>PARSE_TEXT</searchAlgorithm>
                                        <ontologyProperties>
                                            <language>ENG</language>
                                            <outputLanguages>ENG</outputLanguages>
                                            <termSearchType>CONCEPTS_AND_GROUPS</termSearchType>
                                            <returnRelations>false</returnRelations>
                                            <returnGroups>true</returnGroups>
                                        </ontologyProperties>
                                    </action>
                                </xmp:findTerms>
                            </soapenv:Body>
                        </soapenv:Envelope>"""  

    body_expand_terms = """<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xmp="http://xmp.actonomy.com">
                            <soapenv:Body>
                                <xmp:expandTerm>
                                    <action>
                                        <input>{body_text}</input>
                                        <expansionProperties>
                                            <functionExpansionType>ALL</functionExpansionType>
                                            <nrOfResultsSets>1</nrOfResultsSets>
                                        </expansionProperties>
                                        <ontologyProperties>
                                            <language>ENG</language>
                                            <outputLanguages>ENG</outputLanguages>
                                        </ontologyProperties>
                                    </action>
                                </xmp:expandTerm>
                            </soapenv:Body>
                        </soapenv:Envelope>"""


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

    def clean_text_full(self,s):
        # Replace symbols with language
        s = s.replace('&', ' and ')
        s = s.replace('#', ' sharp')
        s = s.replace('@', ' at ')
        s = s.replace('*', ' star ')
        s = s.replace('%', ' percent ')

        s = s.replace('(', ' obr ')
        s = s.replace(')', ' cbr ')
        s = s.replace('{', ' ocbr ')
        s = s.replace('}', ' ccbr ')
        s = s.replace('[', ' osbr ')
        s = s.replace(']', ' csbr ')

        s = s.replace('=', ' equal ')
        s = s.replace('>', ' greater than ')
        s = s.replace('<', ' less than ')
        s = s.replace('+', ' plus ')
        s = s.replace('-', ' dash ')
        s = s.replace('/', ' fsl ')
        s = s.replace('?', ' qm ')
        s = s.replace('!', ' em ')

        s = s.replace('.', ' dot ')
        s = s.replace(',', ' coma ')
        s = s.replace(':', ' colon ')
        s = s.replace(';', ' semicolon ')

        s = re.sub('[^0-9a-zA-Z]+', ' ', s)
        return  s
    

    def send_to_actonomy(self, text, body, row_index, clean_text=True):
        if clean_text:
            encoded_string = self.clean_text_full(text)
        else:
            encoded_string = text

        # replace content keyword in XML template with actual content encoded as base64
        body_subm = body.format(body_text=encoded_string)
        body_subm = body_subm.encode('utf-8')

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
            out_json = self.xml_actonomy_2json(r.text, 'ONT')
        except Exception as e:
            print (e)
            print (str(datetime.now()), 'Error in Actonomy returned XML at dict key: ', row_index)
            return False
        
        return out_json

    
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
        
    def replace_xml_tags(self,xml_tag):
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
                        w_tag = child_tag + '_' + self.clean_text(child_text)

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


    def ontology_json_to_classes(self, groups, labels):
        out_dict = {}
        for i in range(0,len(labels)):
            group = groups[i]
            label = labels[i]
            
            if out_dict.get(group, None) == None:
                out_dict[group] = []
                
            out_dict[group].append(label)
            
        return out_dict

    def actonomy_find_term_fcs(self, ont_json):
        ont_classes = self.ontology_json_to_classes(ont_json.get('ONT_Body_findTermsResponse_return_termHits_groups_termCategory',[]), ont_json.get('ONT_Body_findTermsResponse_return_termHits_groups_preferredLabels_label',[]))  
        return ont_classes

    def actonomy_expand_term_fcs(self, ont_json):
        ont_classes   = self.ontology_json_to_classes(ont_json.get('ONT_Body_expandTermResponse_return_expansions_expansions_term_termCategory',[]), ont_json.get('ONT_Body_expandTermResponse_return_expansions_expansions_term_preferredLabels_label',[]))
        
        ont_skillsets = self.ontology_json_to_classes(ont_json.get('ONT_Body_expandTermResponse_return_expansions_skillSets_term_termCategory',[]), ont_json.get('ONT_Body_expandTermResponse_return_expansions_skillSets_term_preferredLabels_label',[]))
        
        ont_classes.update(ont_skillsets)
        
        ont_abstracts = self.ontology_json_to_classes(ont_json.get('ONT_Body_expandTermResponse_return_expansions_expansions_term_termHits_termCategory',[]), ont_json.get('ONT_Body_expandTermResponse_return_expansions_expansions_term_termHits_preferredLabels_label',[]))
        
        ont_classes.update(ont_abstracts)
        
        ont_classes['SYNONYMS'] = ont_json.get('ONT_Body_expandTermResponse_return_expansions_startTerm_synonymLabels_label',[]) + ont_json.get('ONT_Body_expandTermResponse_return_expansions_expansions_term_termHits_synonymLabels_label',[])
        
        return ont_classes


    def ontology_lists_from_json(self, in_json, json_items, json_item_title, row_index):        
        lst = []
        for item in json_items:
            lst = lst + in_json.get(item, []) 
        features = ','.join(lst)
        
        ont_json    = self.send_to_actonomy(features, self.body_find_terms, row_index, clean_text=True)
        ont_classes = self.actonomy_find_term_fcs(ont_json)
        
        for title in in_json.get(json_item_title, []):
            ont_json = self.send_to_actonomy(title, self.body_expand_terms, row_index, clean_text=True)
            ont_titl = self.actonomy_expand_term_fcs(ont_json)
            
            for key in ont_titl:
                if 'TITLE_ONTOLOGY_' + key not in ont_classes:
                    ont_classes['TITLE_ONTOLOGY_'+key] = ont_titl[key]
                else:
                    ont_classes['TITLE_ONTOLOGY_'+key] = ont_classes['TITLE_ONTOLOGY_'+key] + ont_titl[key]

        return ont_classes


    def run_on(self, df_run, apply_fun=False):
        self.new_columns = []
        new_col_name = self.new_field_prefix + '_' + str(self.result_id)
        self.new_columns.append(new_col_name)

        col_name = self.data_defs[0].split("|")[0]

        try:
            col_name_add = self.data_defs.get[1].split("|")[0]
            col_add_dict = dict( zip(df_run[col_name], df_run['dict_'+col_name_add]) )
        except:
            col_name_add = None
            col_add_dict = None

        col_dict_new = {}

        if apply_fun:
            fld_dict = self.make_dict(df_run['dict'+col_name].fillna(''))             # create dictionary of given text column  
            df_run[new_col_name] = df_run['dict'+col_name].fillna('').map(fld_dict)   # replace column values with corresponding values from dictionary
            col_dict = {v:k for k,v in fld_dict.items()}                              # reverse new dictionary so it can be iterated over keys
        else:
            df_run[new_col_name] = df_run[col_name]    # keys are the same as original column
            col_dict = self.dicts_cols[0]

        block_progress = 0
        index = 0
        total = len(col_dict)
        block = int(total/50)
        
        json_item_title = 'JOB_PositionProfile_PositionDetail_PositionTitle'
        json_items = ['JOB_PositionProfile_PositionDetail_PositionTitle', 'JOB_PositionProfile_PositionDetail_Skills']

        for k,v in col_dict.items():
            row_json = json.loads(v)

            row_json[json_item_title] = row_json.get(json_item_title,[]) + [col_add_dict.get(k, '')]

            j_parsed = json.dumps(self.ontology_lists_from_json(row_json, json_items, json_item_title, k))

            col_dict_new[k] = j_parsed

            block_progress += 1
            index += 1
            if (block_progress >= block):
                block_progress = 0
                print (str(datetime.now()), " keys processed: ", round((index)/total*100,0), "%")

        df_run['dict_' + new_col_name] = df_run[new_col_name].map(col_dict_new)

        return col_dict_new   


    def run(self, mode):
        print ("enter run mode " + str(mode))

        self.dicts_cols = []
        for i in range(0,len(self.data_defs)):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if i==0:
                self.df = pd.read_csv(workdir+file_name)[[col_name]]
            else:
                self.df = self.df.merge(pd.read_csv(workdir+file_name)[[col_name]], left_index=True, right_index=True)

            dsfile = workdir + 'dict_' + file_name
            if os.path.isfile(dsfile):
                # load dictionary if it exists
                dict_temp = pd.read_csv(dsfile, dtype={'value': object}).set_index('key')["value"].to_dict()
                self.dicts_cols.append(dict_temp)
                # replace column with its mapped value from dictionary
                self.df['dict_'+col_name] = self.df[col_name].map(dict_temp)
            else:
                # no dictionary file found
                raise ValueError('Aborting: No dictionary file found for the field: ', col_name)
                return

        fld_dict = self.run_on(self.df)
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
