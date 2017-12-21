if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import numpy as np
    import math
    import re
    import pycountry
    
    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    col_definition2 = "{random_dict}"
    col2 = col_definition2.split("|")[0]
    file2 = col_definition2.split("|")[1]
    result_id = {id}
    field_prefix = 'diffs_'
    
    cn = ['vietnam','laos','america','american','americans','oceania','banglore','chandigarh','pune','french','finnish','israil',
             'israel','usa','europe','european','zealand','swiss','asian','asians','qatar','louisiana','turkey','russia','russian',
             'russians','nato','canada','japan','usa','india','russia','pune','banglore','nato','vietnam','china','ukraine','spain',
             'swiss','france','germany','america','oceania','pakistan','turkey','malaysia','island','french','finnish','massachusetts',
             'laos','baluchistan', 'chad', 'indonesia', 'jamaica', 'indian', 'world', 
             'actinium','aluminium','americium','antimony','argon','arsenic','astatine','barium','berkelium','beryllium','bismuth','bohrium','boron',
             'bromine','cadmium','caesium','calcium','californium','carbon','cerium','chlorine','chromium','cobalt','copernicium','copper','curium',
             'darmstadtium','dubnium','dysprosium','einsteinium','erbium','europium','fermium','flerovium','fluorine','francium','gadolinium','gallium',
             'germanium','gold','hafnium','hassium','helium','holmium','hydrogen','indium','iodine','iridium','iron','krypton','lanthanum','lawrencium',
             'lead','lithium','livermorium','lutetium','magnesium','manganese','meitnerium','mendelevium','mercury','molybdenum','moscovium','neodymium',
             'neon','neptunium','nickel','nihonium','niobium','nitrogen','nobelium','oganesson','osmium','oxygen','palladium','phosphorus','platinum',
             'plutonium','polonium','potassium','praseodymium','promethium','protactinium','radium','radon','rhenium','rhodium','roentgenium','rubidium',
             'ruthenium','rutherfordium','samarium','scandium','seaborgium','selenium','silicon','silver','sodium','strontium','sulfur','tantalum','technetium',
             'tellurium','tennessine','terbium','thallium','thorium','thulium','tin','titanium','tungsten','uranium','vanadium','xenon','ytterbium','yttrium',
             'zinc','zirconium','republic','switzerland','baroda','london','americas','canadian','alaska','columbia','usb']
    
    sw = set([u'i', u"i'm", u'me', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn',  u'wasn', u'weren', u'won', u'wouldn', u'would'])

    mapping_case = [
        [' US ', ' USA '],
        [' US.', ' USA.'],
        [' US,', ' USA,'],
        [' US?', ' USA?'],
        [' US)', ' USA)'],
        ['MMORPG', 'game'],
        ['United States','USA'],
        ['united states','USA'],
        ['OITNB','oitnb']
    ]

    mapping = [
        ['&',''],
        ["'",''],
        ['u.s.', 'usa '],
        ['united states', 'usa'],
        ['orange is the new black', 'oitnb'],
        ['year old', ''],
        ['years old', ''],
        ['year olds', ''],
        ['years olds', ''],
        ['most effective', 'easiest'],
        ['least painful', 'easiest'],
        ['how much', 'what is'],
        ["what's", 'what is'],
        ['black money','corruption'],
        [' s & t ',' sandt '],
        [' s&t ',' sandt '],
        [' m & t ',' mandt '],
        [' m&t ',' mandt '],
        ['non-schedule','nonschedule'],
        ['lakeland financial','lakelandfinancial'],
        ["banksy's",'banksy'],
        [' at&t ','atandt'],
        [' at & t ','atandt']
    ]

    mapping_word = [
        ['parallelisms','relation'],
        ['parallelism','relation'],
        ['toddler', 'children'],
        ['child', 'children'],
        ['baby', 'children'],
        ['top', 'best'],
        ['cheapest', 'easiest'],
        ['teenager', 'teen'],
        ['teenage', 'teen'],
        ['movy', 'movie'],
        ['which', 'what']
    ]
    
    def func(self, s):
        return s[:1].lower() + s[1:] if s else ''
    
    def __init__(self):
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        self.fldprefix = self.field_prefix + str(self.result_id)
        
        for c in list(self.pycountry.countries):
            cname = c.name.lower()
            self.cn.append(cname)
            self.cn.append(cname + 'n')
        
        
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        if self.col2 not in dicts:
            self.dict2 = self.pd.read_csv(workdir+'dict_'+self.col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict2 = {v:k for k,v in dicts[self.col2].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = self.df[self.col1].map(self.dict1)
        self.dfx[self.col2] = self.df[self.col2].map(self.dict2)
        
        
        print ( "calculating word counts" )
        self.word_counts = {}

        block = len(self.dfx)/50
        k=block-1
        for index, row in self.dfx.iterrows():
            k+=1
            if type(row[self.col1])==str:
                sline1 = self.replacements(row[self.col1])
            else:
                sline1 = ''
            if type(row[self.col2])==str:
                sline2 = self.replacements(row[self.col2])
            else:
                sline2 = ''

            wta = self.replacements_word(self.re.findall(r"[\w']+", sline1))
            wtb = self.replacements_word(self.re.findall(r"[\w']+", sline2))

            for w in wta+wtb:
                if w in self.word_counts:
                    self.word_counts[w]+=1
                else:
                    self.word_counts[w]=1
            if k>=block:
                print (index)
                k=0

        print ( "ok" )

    def replacements(self, sline):
        res = sline
        for a in self.mapping_case:
            res = res.replace(a[0], a[1])
        res = res.lower()
        for a in self.mapping:
            res = res.replace(a[0], a[1])
        return res

    def replacements_word(self, aword):
        res = list(aword)
        for i in range(len(res)):
            head, middle, tail = res[i].rpartition('ies')
            if len(tail)==0:
                res[i] = head+'y'
            for a in self.mapping_word:
                if res[i]==a[0]:
                    res[i] = a[1]
        return res

    def get_diffs2(self, sline1, sline2):
        wta = self.replacements_word(self.re.findall(r"[\w']+", sline1.lower()))
        wtb = self.replacements_word(self.re.findall(r"[\w']+", sline2.lower()))

        swta = set(wta)
        swtb = set(wtb)
        wta2 = list(swta-swtb)
        wtb2 = list(swtb-swta)

        wta3 = list(set(wta2)-self.sw)
        wtb3 = list(set(wtb2)-self.sw)

        cna = set(wta3).intersection(self.cn)
        cnb = set(wtb3).intersection(self.cn)

        wta3 = list(set(wta3)-cna)
        wtb3 = list(set(wtb3)-cnb)

        common1 = len(swta)-len(wta2)
        common2 = len(swtb)-len(wtb2)

        for w1 in list(wta3):
            for w2 in list(wtb3):
                if w1==w2+'s' or w1+'s'==w2 or w1==w2+'ly' or w1+'ly'==w2:
                    try:
                        wta3.remove(w1)
                    except:
                        pass
                    try:
                        wtb3.remove(w2)
                    except:
                        pass
                    common1+=1
                    common2+=1

        sum1=0
        sum2=0
        numbers1=0
        numbers2=0
        for i in range(len(wta3)):
            if wta3[i] not in self.word_counts:
                self.word_counts[wta3[i]]=1
            sum1+=int(1000000.0/(1+self.word_counts[wta3[i]]))
            try:
                numbers1+=float(wta3[i])
            except:
                pass
        for i in range(len(wtb3)):
            if wtb3[i] not in self.word_counts:
                self.word_counts[wtb3[i]]=1
            sum2+=int(1000000.0/(1+self.word_counts[wtb3[i]]))
            try:
                numbers2+=float(wtb3[i])
            except:
                pass

        return common1, common2, wta3, wtb3, list(cna), list(cnb), sum1, sum2, numbers1, numbers2

    def run_on(self, df_run):
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
        if self.col2 not in dicts:
            self.dict2 = self.pd.read_csv(workdir+'dict_'+self.col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict2 = {v:k for k,v in dicts[self.col2].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)
        self.dfx[self.col2] = df_run[self.col2].map(self.dict2)

        block = int(len(df_run)/50)
        i = 0

        for index, row in self.dfx.iterrows():
            i+=1
            if type(row[self.col1])==str:
                sline1 = self.func(row[self.col1])
            else:
                sline1 = ''
            if type(row[self.col2])==str:
                sline2 = self.func(row[self.col2])
            else:
                sline2 = ''


            ncomm1, ncomm2, a1, a2, cna, cnb, sum1, sum2, n1, n2 = self.get_diffs2(self.replacements(sline1), self.replacements(sline2))

            df_run.set_value(index, self.fldprefix + '_1', ncomm1) # count of common words
            df_run.set_value(index, self.fldprefix + '_2', len(a1)) # count of differ words in sentance 1
            df_run.set_value(index, self.fldprefix + '_3', len(a2)) # count of differ words in sentance 2
            df_run.set_value(index, self.fldprefix + '_4', len(a1)+len(a2)) # count of differ words in sentance 2
            df_run.set_value(index, self.fldprefix + '_5', len(cna)) # count of differ countries in sentance 1
            df_run.set_value(index, self.fldprefix + '_6', len(cnb)) # count of differ countries in sentance 2
            df_run.set_value(index, self.fldprefix + '_7', len(cna)+len(cnb))
            df_run.set_value(index, self.fldprefix + '_8', sum1) # total differ words importance of sentance 1
            df_run.set_value(index, self.fldprefix + '_9', sum2) # total differ words importance of sentance 2
            df_run.set_value(index, self.fldprefix + '_10', abs(sum1-sum2))
            df_run.set_value(index, self.fldprefix + '_11', n1) # sum of numbers in sentance 1
            df_run.set_value(index, self.fldprefix + '_12', n2) # sum of numbers in sentance 2
            df_run.set_value(index, self.fldprefix + '_13', abs(n1-n2))


            if i>=block and block>=1000:
                i=0
                print (index)


        df_run[[self.fldprefix + '_11',self.fldprefix + '_12',self.fldprefix + '_13']]=df_run[[self.fldprefix + '_11',self.fldprefix + '_12',self.fldprefix + '_13']].fillna(value=0)

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.run_on(self.df)
        
        nrow = len(self.df)

        for i in range(1,14):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
