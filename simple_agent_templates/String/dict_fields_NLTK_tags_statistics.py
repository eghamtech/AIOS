#start_of_parameters
#key=fields_source;  type=constant;  value=['dict_field|dict_field.csv','dict_field1|dict_field1.csv','dict_field2|dict_field2.csv']
#key=col_max_length;  type=constant;  value=200
#key=new_field_prefix;  type=constant;  value=nltk_pos_stats_
#key=include_columns_type;  type=constant;  set=is_dict_only
#key=include_columns_containing;  type=constant;  set=
#key=ignore_columns_containing;  type=constant;  set='%ev_field%' and '%onehe_%'
#end_of_parameters

# AICHOO OS Simple Agent
# Documentation about AI OS and how to create Simple Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Simple-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction
#
# this agent creates new columns from given fields by tokenizing concatenated text and tagging all tokens
# all source fields expected to be dictionary fields
#
# number of new columns created will be the same as number of unique tags in pos_tags map 
# plus two columns with number of tokens and number of words
#
# if "fields_source" parameter not specified then 2 fields will be obtained randomly
# according to normal AIOS logic

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import nltk
import os.path, bz2, pickle, re

from datetime import datetime
from collections import Counter

class cls_agent_{id}:

    data_defs = {fields_source}

    # obtain a unique ID for the current instance
    result_id         = {id}
    # create new field name based on "new_field_prefix" with unique instance ID
    # and filename to save new field data
    new_field_prefix  = "{new_field_prefix}"
    col_max_length    = {col_max_length}
    agent_name        = 'agent_' + str(result_id)

    dicts_agent = {}
    new_columns = []
    dict_cols   = []
    
    # All Part of Speech tags to be identified and counted
    pos_tags = {
        "$"  : 'DLR',  # dollar
                       # $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
        "''" : 'QM',   # closing quotation mark  ' ''
        '('  : 'OB',   # opening parenthesis ( [ {
        ')'  : 'CB',   # closing parenthesis ) ] }
        ','  : 'COMA', # comma ,
        '--' : 'DASH', # dash --
        '.'  : 'DOT',  # sentence terminator . ! ?
        ':'  : 'CLN',  # colon or ellipsis : ; ...
        'CC' : 'CC',   # conjunction, coordinating
                       # & 'n and both but either et for less minus neither nor or plus so
                       # therefore times v. versus vs. whether yet
        'CD' : 'CD',   # numeral, cardinal
                       # mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
                       # seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
                       # fifteen 271,124 dozen quintillion DM2,000 ...
        'DT' : 'DT',   # determiner
                       # all an another any both del each either every half la many much nary
                       # neither no some such that the them these this those
        'EX' : 'EX',   # existential there there
        'FW' : 'FW',   # foreign word
                       # gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
                       # lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
                       # terram fiche oui corporis ...
        'IN' : 'IN',   # preposition or conjunction, subordinating
                       # astride among uppon whether out inside pro despite on by throughout
                       # below within for towards near behind atop around if like until below
                       # next into if beside ...
        'JJ' : 'JJ',   # adjective or numeral, ordinal
                       # third ill-mannered pre-war regrettable oiled calamitous first separable
                       # ectoplasmic battery-powered participatory fourth still-to-be-named
                       # multilingual multi-disciplinary ...
        'JJR': 'JJR',  # adjective, comparative
                       # bleaker braver breezier briefer brighter brisker broader bumper busier
                       # calmer cheaper choosier cleaner clearer closer colder commoner costlier
                       # cozier creamier crunchier cuter ...
        'JJS': 'JJS',  # adjective, superlative
                       # calmest cheapest choicest classiest cleanest clearest closest commonest
                       # corniest costliest crassest creepiest crudest cutest darkest deadliest
                       # dearest deepest densest dinkiest ...
        'LS' : 'LS',   # list item marker
                       # A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
                       # SP-44007 Second Third Three Two * a b c d first five four one six three two
        'MD' : 'MD',   # modal auxiliary
                       # can cannot could couldn't dare may might must need ought shall should
                       # shouldn't will would
        'NN' : 'NN',   # noun, common, singular or mass
                       # common-carrier cabbage knuckle-duster Casino afghan shed thermostat
                       # investment slide humour falloff slick wind hyena override subhumanity
                       # machinist ...
        'NNP': 'NNP',  # noun, proper, singular
                       # Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
                       # Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
                       # Shannon A.K.C. Meltex Liverpool ...
        'NNPS':'NNPS', # noun, proper, plural
                       # Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
                       # Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
                       # Apache Apaches Apocrypha ...
        'NNS':'NNS',   # noun, common, plural
                       # undergraduates scotches bric-a-brac products bodyguards facets coasts
                       # divestitures storehouses designs clubs fragrances averages
                       # subjectivists apprehensions muses factory-jobs ...
        'PDT':'PDT',   # pre-determiner
                       # all both half many quite such sure this
        'POS':'POS',   # genitive marker ' 's
        'PRP':'PRP',   # pronoun, personal
                       # hers herself him himself hisself it itself me myself one oneself ours
                       # ourselves ownself self she thee theirs them themselves they thou thy us
        "PRP$":'PRPD', # pronoun, possessive
                       # her his mine my our ours their thy your
        'RB' : 'RB',   # adverb
                       # occasionally unabatingly maddeningly adventurously professedly
                       # stirringly prominently technologically magisterially predominately
                       # swiftly fiscally pitilessly ...
        'RBR': 'RBR',  # adverb, comparative
                       # further gloomier grander graver greater grimmer harder harsher
                       # healthier heavier higher however larger later leaner lengthier less-
                       # perfectly lesser lonelier longer louder lower more ...
        'RBS': 'RBS',  # adverb, superlative
                       # best biggest bluntest earliest farthest first furthest hardest
                       # heartiest highest largest least less most nearest second tightest worst
        'RP' : 'RP',   # particle
                       # aboard about across along apart around aside at away back before behind
                       # by crop down ever fast for forth from go high i.e. in into just later
                       # low more off on open out over per pie raising start teeth that through
                       # under unto up up-pp upon whole with you
        'SYM': 'SYM',  # symbol
                       # % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
        'TO' : 'TO',   # "to" as preposition or infinitive marker to
        'UH' : 'UH',   # interjection
                       # Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
                       # huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
                       # man baby diddle hush sonuvabitch ...
        'VB' : 'VB',   # verb, base form
                       # ask assemble assess assign assume atone attention avoid bake balkanize
                       # bank begin behold believe bend benefit bevel beware bless boil bomb
                       # boost brace break bring broil brush build ...
        'VBD': 'VBD',  # verb, past tense
                       # dipped pleaded swiped regummed soaked tidied convened halted registered
                       # cushioned exacted snubbed strode aimed adopted belied figgered
                       # speculated wore appreciated contemplated ...
        'VBG': 'VBG',  # verb, present participle or gerund
                       # telegraphing stirring focusing angering judging stalling lactating
                       # hankerin' alleging veering capping approaching traveling besieging
                       # encrypting interrupting erasing wincing ...
        'VBN': 'VBN',  # verb, past participle
                       # multihulled dilapidated aerosolized chaired languished panelized used
                       # experimented flourished imitated reunifed factored condensed sheared
                       # unsettled primed dubbed desired ...
        'VBP': 'VBP',  # verb, present tense, not 3rd person singular
                       # predominate wrap resort sue twist spill cure lengthen brush terminate
                       # appear tend stray glisten obtain comprise detest tease attract
                       # emphasize mold postpone sever return wag ...
        'VBZ': 'VBZ',  # verb, present tense, 3rd person singular
                       # bases reconstructs marks mixes displeases seals carps weaves snatches
                       # slumps stretches authorizes smolders pictures emerges stockpiles
                       # seduces fizzes uses bolsters slaps speaks pleads ...
        'WDT': 'WDT',  # WH-determiner
                       # that what whatever which whichever
        'WP' : 'WP',   # WH-pronoun
                       # that what whatever whatsoever which who whom whosoever
        "WP$": 'WPD',  # WH-pronoun, possessive whose
        'WRB': 'WRB',  # Wh-adverb
                       # how however whence whenever where whereby whereever wherein whereof why
        "``" : 'OQM',  # opening quotation mark ` ``
        'NAN': 'NAN'   # Unknown Tag
    }
    

    def is_set(self, s):
        return len(s)>0 and s!="0"

    def make_dict(self, col):
        a1 = col.unique()
        a1 = [x for x in a1 if str(x) != 'nan']
        keys1 = range(1, len(a1)+1)
        return dict(zip(a1, keys1))

    def __init__(self):
        #if not self.is_set(self.data_defs):
        #    self.data_defs = ["{_random_dict_distinct}", "{_random_dict_distinct}"]

        # if saved dictionaries for the target field already exist then load them from filesystem
        sfile = workdir + self.agent_name + '.model'
        if os.path.isfile(sfile):
            rfile = bz2.BZ2File(sfile, 'r')
            self.dicts_agent = pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.agent_name + ': NLTK POS Tags Stats agent dictionaries model loaded')


    def run_on(self, df_run, apply_fun=False):
        self.new_columns = []
        
        stop_words = set(nltk.corpus.stopwords.words('english'))

        # commented out because new data should already come with 'dict_' colums used by this agent
        # if apply_fun:
        #     for col_name in self.dicts_agent['dict_cols']:
        #         df_run['dict_'+col_name] = df_run[col_name]   # .map( self.dicts_agent[col_name] )   - new data should come as text, not dictionary key

        new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_NumTokens'
        new_col_name = new_col_name[:self.col_max_length]
        df_run[new_col_name] = 0
        self.new_columns.append(new_col_name)
        
        new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_NumWords'
        new_col_name = new_col_name[:self.col_max_length]         
        df_run[new_col_name] = 0
        self.new_columns.append(new_col_name)
            
        for k,v in self.dicts_agent['pos_dicts'].items():
            # all Part of Speech Tag Names should be stored in this dictionary as values, so just iterate over them
            new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_' + re.sub('[^0-9a-zA-Z]+', '_', str(v))
            new_col_name = new_col_name[:self.col_max_length]
            self.new_columns.append(new_col_name)
            df_run[new_col_name] = 0

        
        block_progress = 0
        total = len(df_run)
        block = int(total/20)
        
        for index, row in df_run.iterrows():
            row_str = ''
            for col_name in self.dicts_agent['dict_cols']:
                row_str += ' ' + str(row['dict_'+col_name])   # concatenate columns into one string

            row_str = row_str[1:]                             # remove space added during columns concatenation
            
            # tokenize and tag tokens
            row_tokens = nltk.tokenize.word_tokenize(row_str)   
            row_pstags = nltk.tag.pos_tag(row_tokens)
            
            # count tokens and create new column for it
            new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_NumTokens'
            new_col_name = new_col_name[:self.col_max_length]
            df_run.at[index, new_col_name] = len(row_tokens)
            
            # count tokens without stop words and create new column for it
            new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_NumWords'
            new_col_name = new_col_name[:self.col_max_length]         
            row_words    = [w for w in row_tokens if not w in stop_words]
            df_run.at[index, new_col_name] = len(row_words)
            
            # count Part of Speech tags and save the number in corresponding column
            row_pstags_counts = Counter( tag for word, tag in row_pstags)
            for tag in row_pstags_counts:
                tag_str      = self.dicts_agent['pos_dicts'].get(tag,'NAN')
                new_col_name = self.new_field_prefix + '_' + str(self.result_id) + '_v_' + re.sub('[^0-9a-zA-Z]+', '_', str(tag_str))
                new_col_name = new_col_name[:self.col_max_length]
                
                df_run.at[index, new_col_name] = row_pstags_counts[tag]

            block_progress += 1
            if (block_progress >= block):
                block_progress = 0
                print (str(datetime.now()), " rows processed: ", round((index+1)/total*100,0), "%")
                    

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

                #self.dicts_agent[col_name] = dict_temp
                self.df['dict_'+col_name]  = self.df[col_name].map(dict_temp)

                self.dict_cols.append(col_name)


        self.dicts_agent['dict_cols'] = self.dict_cols
        self.dicts_agent['pos_dicts'] = self.pos_tags

        self.run_on(self.df)
        nrow = len(self.df)

        self.dicts_agent['new_columns'] = self.new_columns
        # save dictionary of all auxiliary data into file
        sfile = bz2.BZ2File(workdir + self.agent_name + '.model', 'w')
        pickle.dump(self.dicts_agent, sfile)
        sfile.close()

        # save and register each new column
        for i in range(0,len(self.new_columns)):
            fld   = self.new_columns[i]
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))


    def apply(self, df_add):
        self.run_on(df_add, apply_fun=True)

agent_{id} = cls_agent_{id}()
