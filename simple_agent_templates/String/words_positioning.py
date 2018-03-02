if 'dicts' not in globals():
    dicts = {}

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    result_id = {id}
    field_prefix = 'wpos_' + col1 + '_'

    whats = ['why', 'how', 'what', 'where', 'when', 'which', 'who', 'whom']

    sw = set([u'i', u"i'm", u'me', u'myself', u'my', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its',
     u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has',
     u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into',
     u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there',  
     u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just',
     u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn',
     u'wasn', u'weren', u'won', u'wouldn', u'would'])

    replacements_case = {' US ':' USA ', ' US.':' USA.', ' US,':' USA,', ' US?':' USA?', ' US)':' USA)', 'MMORPG':'game', 'United States':'USA', 'united states':'USA', 'OITNB':'oitnb'}
    replacements = {'from scratch':'','u.s.':'usa ','united states':'usa','orange is the new black':'oitnb','year old':'',
        'years old':'','year olds':'','years olds':'','most effective':'easiest','least painful':'easiest','how much':'what is',
        "what's":'what is','black money':'corruption',' s & t ':' sandt ',' s&t ':' sandt ',' m & t ':' mandt ',' m&t ':' mandt ',
        'non-schedule':'nonschedule','lakeland financial':'lakelandfinancial',"banksy's":'banksy',' at&t ':'atandt',' at & t ':'atandt'}
    
    anto={'boyfriend':'girlfriend', 'ipad':'iphone', 'increase':'decrease', 'antichrist':'christ', 'bad':'good', 'best':'worst', 'girls':'guy', 'questions':'answers', 'men':'women', 'guys':'girl',
         'girl':'guys', 'ask':'answer', 'question':'answer'}

    syno=['best___good','good___best','get___getting','getting___get','indian___india','india___indian','use___using','using___use',
          'earn___make','make___earn','best___better','better___best','get___find','find___get','best___top','top___best',
          'different___difference','difference___different','good___better','better___good','making___make','make___making',
          'people___someone','someone___people','best___ever','ever___best','prepare___preparation','preparation___prepare',
          'someone___person','person___someone','get___take','take___get','used___using','using___used','live___living',
          'living___live','best___easiest','easiest___best','people___person','person___people','get___buy','buy___get',
          'like___love','love___like','first___best','best___first','rupee___rs','rs___rupee','use___make','make___use',
          'best___possible','possible___best','someone___one','one___someone','writing___write','write___writing','reduce___lose',
          'lose___reduce','differ___difference','difference___differ','best___great','great___best','engineers___engineering',
          'engineering___engineers','get___got','got___get','phone___mobile','mobile___phone',
          'take___taking','taking___take','development___developer','developer___development',
          'fat___weight','weight___fat','people___others','others___people','prepare___preparing','preparing___prepare',
          'one___person','person___one','questions___answer','answer___questions','lose___loose','loose___lose','job___work',
          'work___job','good___great','great___good','online___internet','internet___online','best___favorite','favorite___best',
          'women___girls','girls___women','preparing___preparation','preparation___preparing','president___presidency',
          'presidency___president','losing___lose','lose___losing','indians___india','india___indians','studies___studying',
          'studying___studies','learn___know','know___learn','made___make','make___made','become___becoming','improve___increase',
          'increase___improve','becoming___become','best___right','country___india','india___country','important___importance',
          'right___best','importance___important','possible___way','way___possible','create___make','make___create',
          'program___programming','programming___program','someone___friend','friend___someone','people___indian','indian___people',
          'marry___married','married___marry','usa___america','america___usa','american___usa','usa___american','ban___banning',
          'banning___ban','best___fastest','fastest___best','differ___different','different___differ','best___high','high___best',
          'rs___rupees','rupees___rs','investment___invest','invest___investment','similar___like','like___similar',
          'ask___question','question___ask','phone___android','android___phone','film___movie','movie___film','get___earn',
          'earn___get','us___usa','usa___us','website___site','site___website','girl___someone','someone___girl','rs___inr',
          'inr___rs','run___running','iii___3','3___iii','running___run','two___2','2___two','questions___ask','ask___questions',
          'woman___women','women___woman','favorite___favourite','favourite___favorite','ask___post','different___differences',
          'differences___different','post___ask','website___online','online___website','best___fast','fast___best',
          'phone___smartphone','smartphone___phone','search___google','google___search','exam___preparation','preparation___exam',
          'country___usa','usa___country','currency___rupee','rupee___currency','u___usa','usa___u','marks___score','score___marks',
          'science___engineering','engineering___science','win___winning','winning___win','email___account','account___email',
          'travel___travelling','india___modi','travelling___travel','rs___notes','effect___affect','affect___effect']
    
    import pycountry
    
    countries = ['vietnam','laos','america','american','americans','oceania','banglore','chandigarh','pune','french','finnish','israil',
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
    
    nfirst = 6
    nlast = 6
    nall = 20
    nextra = 6

    allwords = {}
    istrain = True
    
    def func(self, s):
        return s[:1].lower() + s[1:] if s else ''
    
    def __init__(self):
        self.fldprefix = self.field_prefix + str(self.result_id)
        
        for c in list(self.pycountry.countries):
            cname = c.name.lower()
            self.countries.append(cname)
            self.countries.append(cname + 'n')
            self.countries.append(cname + 'ian')

        #preload words
        
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = self.df[self.col1].map(self.dict1)
        
        print ( "preload words" )

        block = len(self.dfx)/50
        k=block-1
        for index, row in self.dfx.iterrows():
            k+=1
            if type(row[self.col1])==str:
                sline1 = row[self.col1]
            else:
                sline1 = ''

            for key in self.replacements_case.keys():
                sline1 = sline1.replace(key, self.replacements_case[key])

            sline1 = sline1.lower()
    
            for key in self.replacements.keys():
                sline1 = sline1.replace(key, self.replacements[key])
        
            wta = self.my_tokenize(sline1)

            for w1 in wta:
                if w1 not in self.allwords:
                    self.allwords[w1] = len(self.allwords)+1

            if k>=block:
                print (index)
                k=0
        print ("done")
        
    def my_tokenize(self, text):
        filter1 = ',.;`-+#"!?:[]()%$=/' + "'*@~^&|"
        for ch in filter1:
            text = text.replace(ch, ' '+ch)
        return text.strip().split()

    def run_on(self, df_run):
        if self.col1 not in dicts:
            self.dict1 = self.pd.read_csv(workdir+'dict_'+self.col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
        else:
            self.dict1 = {v:k for k,v in dicts[self.col1].items()} # make key=number, value=string
            
        self.dfx = self.pd.DataFrame()
        self.dfx[self.col1] = df_run[self.col1].map(self.dict1)

        block = int(len(df_run)/500)
        k = 0

        for index, row in self.dfx.iterrows():
            k+=1
            if type(row[self.col1])==str:
                sline1 = row[self.col1]
            else:
                sline1 = ''

            for key in self.replacements_case.keys():
                sline1 = sline1.replace(key, self.replacements_case[key])

            wtac = self.my_tokenize(self.func(sline1))

            for w1 in list(wtac):
                if (not w1[0].isupper()) or (w1.lower() in self.whats) or (w1.lower() in self.sw):
                    wtac.remove(w1)

            for i in range(len(wtac)):
                wtac[i] = wtac[i].lower()

            common_caps = set(wtac)
            diff_caps1 = list(set(wtac))

            sline1 = sline1.lower()

            for key in self.replacements.keys():
                sline1 = sline1.replace(key, self.replacements[key])

            wta = self.my_tokenize(sline1)

            idfa=[0]*self.nfirst
            for w1 in wta:
                for i in range(self.nfirst):
                    if idfa[i]==0:
                        if w1 in self.allwords:
                            idfa[i]=self.allwords[w1]
                            break

            idla=[0]*self.nlast
            for w1 in reversed(wta):
                for i in range(self.nlast):
                    if idla[i]==0:
                        if w1 in self.allwords:
                            idla[i]=self.allwords[w1]
                            break

            nworda = ('word' in wta) + 0

            wtpa = []
            for i in range(len(wta)-1):
                wtpa.append(wta[i]+' '+wta[i+1])
            wt3a = []
            for i in range(len(wta)-2):
                wt3a.append(wta[i]+' '+wta[i+1]+' '+wta[i+2])

            common2 = len(set(wtpa))
            common3 = len(set(wt3a))

            common0 = len(set(wta))

            wh1 = set(wta).intersection(self.whats)
            wta = list(set(wta)-wh1)
            diff_whats = len(wh1)

            wta = list(set(wta)-self.sw)

            swta = set(wta)
            wta2 = list(swta)

            wta3 = list(set(wta2))

            common1 = len(swta)

            diff_words1 = len(wta3)


            cn1 = set(wta3).intersection(self.countries)
            wta3 = list(set(wta3)-cn1)
            diff_countries = len(cn1)

            nota = ('not' in wta3) + 0

            idxa=[0]*self.nextra
            for w1 in wta3:
                for i in range(self.nextra):
                    if idxa[i]==0:
                        if w1 in self.allwords:
                            idxa[i]=self.allwords[w1]
                            break

            ida=[0]*self.nall
            for w1 in wta:
                for i in range(self.nall):
                    if ida[i]==0:
                        if w1 in self.allwords:
                            ida[i]=self.allwords[w1]
                            break
            
            df_run.set_value(index, self.fldprefix + '_1', common0)
            df_run.set_value(index, self.fldprefix + '_2', common2)
            df_run.set_value(index, self.fldprefix + '_3', common3)
            df_run.set_value(index, self.fldprefix + '_4', common1)
            df_run.set_value(index, self.fldprefix + '_5', diff_whats)
            df_run.set_value(index, self.fldprefix + '_6', diff_countries)
            df_run.set_value(index, self.fldprefix + '_7', diff_whats+diff_countries)
            df_run.set_value(index, self.fldprefix + '_8', nota)
            df_run.set_value(index, self.fldprefix + '_9', nworda)
            df_run.set_value(index, self.fldprefix + '_10', len(common_caps))
            df_run.set_value(index, self.fldprefix + '_11', len(diff_caps1))
            col = 12
            for i in range(self.nfirst):
                df_run.set_value(index, self.fldprefix + '_'+str(col), idfa[i])
                col+=1
            for i in range(self.nlast):
                df_run.set_value(index, self.fldprefix + '_'+str(col), idla[i])
                col+=1
            for i in range(self.nall):
                df_run.set_value(index, self.fldprefix + '_'+str(col), ida[i])
                col+=1
            for i in range(self.nextra):
                df_run.set_value(index, self.fldprefix + '_'+str(col), idxa[i])
                col+=1

            if k>=block and block>=100:
                k=0
                print (index)

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.run_on(self.df)
        
        total_cols = 11+(self.nfirst+self.nlast+self.nall+self.nextra)

        nrow = len(self.df)
        for i in range(1,total_cols+1):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
