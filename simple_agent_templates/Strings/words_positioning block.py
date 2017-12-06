import pandas as pd

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}

field_prefix = 'wpos_'

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)


##################################################

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


import pycountry
countries = []
for c in list(pycountry.countries):
    cname = c.name.lower()
    countries.append(cname)
    countries.append(cname + 'n')
    countries.append(cname + 'ian')

countries+=['vietnam','laos','america','american','americans','oceania','banglore','chandigarh','pune','french','finnish','israil',
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
     'zinc','zirconium','republic','switzerland','baroda','london','americas','canadian','alaska','columbia']


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


def my_tokenize(text):
    filter1 = ',.;`-+#"!?:[]()%$=/\'*@~^&|'
    for ch in filter1:
        text = text.replace(ch, ' '+ch)
    return text.strip().split()

nfirst = 6
nlast = 6
nall = 20
nextra = 6

allwords = {}
allpairs = {}
istrain = True


fldprefix = field_prefix + str(result_id)

block = int(len(df)/500)
k = block-1

for index, row in df.iterrows():
    k+=1
    if type(row[col1])==str:
        sline1 = row[col1]
    else:
        sline1 = ''
    if type(row[col2])==str:
        sline2 = row[col2]
    else:
        sline2 = ''

    for key in replacements_case.keys():
        sline1 = sline1.replace(key, replacements_case[key])
        sline2 = sline2.replace(key, replacements_case[key])
    
    func = lambda s: s[:1].lower() + s[1:] if s else ''
    wtac = my_tokenize(func(sline1))
    wtbc = my_tokenize(func(sline2))
    
    for w1 in list(wtac):
        if (not w1[0].isupper()) or (w1.lower() in whats) or (w1.lower() in sw):
            wtac.remove(w1)


    for w1 in list(wtbc):
        if (not w1[0].isupper()) or (w1.lower() in whats) or (w1.lower() in sw):
            wtbc.remove(w1)

    for i in range(len(wtac)):
        wtac[i] = wtac[i].lower()
    for i in range(len(wtbc)):
        wtbc[i] = wtbc[i].lower()

    common_caps = set(wtac).intersection(set(wtbc))
    diff_caps1 = list(set(wtac)-set(wtbc))
    diff_caps2 = list(set(wtbc)-set(wtac))
    
    sline1 = sline1.lower()
    sline2 = sline2.lower()
    
    for key in replacements.keys():
        sline1 = sline1.replace(key, replacements[key])
        sline2 = sline2.replace(key, replacements[key])
    

        
    wta = my_tokenize(sline1)
    wtb = my_tokenize(sline2)
    
    
    
    idfa=[0]*nfirst
    for w1 in wta:
        for i in range(nfirst):
            if idfa[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    idfa[i]=allwords[w1]
                    break
    
    idfb=[0]*nfirst
    for w1 in wtb:
        for i in range(nfirst):
            if idfb[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    idfb[i]=allwords[w1]
                    break
    idla=[0]*nlast
    for w1 in reversed(wta):
        for i in range(nlast):
            if idla[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    idla[i]=allwords[w1]
                    break
    
    idlb=[0]*nlast
    for w1 in reversed(wtb):
        for i in range(nlast):
            if idlb[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    idlb[i]=allwords[w1]
                    break
                    
    nworda = ('word' in wta) + 0
    nwordb = ('word' in wtb) + 0

    
    wtpa = []
    for i in range(len(wta)-1):
        wtpa.append(wta[i]+' '+wta[i+1])
    wtpb = []
    for i in range(len(wtb)-1):
        wtpb.append(wtb[i]+' '+wtb[i+1])
    wt3a = []
    for i in range(len(wta)-2):
        wt3a.append(wta[i]+' '+wta[i+1]+' '+wta[i+2])
    wt3b = []
    for i in range(len(wtb)-2):
        wt3b.append(wtb[i]+' '+wtb[i+1]+' '+wtb[i+2])
    
    common2 = len(set(wtpa).intersection(set(wtpb)))
    common3 = len(set(wt3a).intersection(set(wt3b)))
    
    
    common0 = len(set(wta).intersection(set(wtb)))
    total_words0 = len(set(wta).union(set(wtb)))

    wh1 = set(wta).intersection(whats)
    wh2 = set(wtb).intersection(whats)
    wta = list(set(wta)-wh1)
    wtb = list(set(wtb)-wh2)
    diff_whats = len(wh1.union(wh2))
    
    wta = list(set(wta)-sw)
    wtb = list(set(wtb)-sw)
    
    
    
    
    
    
    swta = set(wta)
    swtb = set(wtb)
    wta2 = list(swta-swtb)
    wtb2 = list(swtb-swta)

    wta3 = list(set(wta2))
    wtb3 = list(set(wtb2))

    common1 = len(swta.intersection(swtb))
    total_words1 = len(swta.union(swtb))
    
    diff_words1 = len(wta3)+len(wtb3)
    
    repeats1 = 0
    def cleanup_wta3(w1, w2):
        try:
            wta3.remove(w1)
            wtb3.remove(w2)
        except:
            pass
    
    
    for w1 in list(wta3):
        for w2 in list(wtb3):
            if w1==w2+'s' or w1==w2+'es' or w1==w2+'ing' or w1==w2+'ed' or w1==w2+'d' or w1+'ly'==w2 or (w1[:-1]==w2[:-3] and w1[-1:]=='y' and w2[-3:]=='ies'):
                repeats1+=2
                cleanup_wta3(w1,w2)
            elif w2==w1+'s' or w2==w1+'es' or w2==w1+'ing' or w2==w1+'ed' or w2==w1+'d' or w2+'ly'==w1 or (w2[:-1]==w1[:-3] and w2[-1:]=='y' and w1[-3:]=='ies'):
                repeats1+=2
                cleanup_wta3(w1,w2)
            else:
                pass
            
    
    diff_words2 = len(wta3)+len(wtb3)
    
    synonyms1 = 0
    for w1 in list(wta3):
        for w2 in list(wtb3):
            if ((w1+'___'+w2) in syno) or ((w2+'___'+w1) in syno):
                synonyms1+=2
                cleanup_wta3(w1,w2)
    
    diff_words3 = len(wta3)+len(wtb3)
    

    
    cn1 = set(wta3).intersection(countries)
    cn2 = set(wtb3).intersection(countries)
    wta3 = list(set(wta3)-cn1)
    wtb3 = list(set(wtb3)-cn2)
    diff_countries = len(cn1.union(cn2))
    
    nants = 0
    for w1 in wta3:
        for w2 in wtb3:
            if w1 in anto:
                if anto[w1]==w2:
                    nants+=2
            elif w2 in anto:
                if anto[w2]==w1:
                    nants+=2
    
    nota = ('not' in wta3) + 0
    notb = ('not' in wtb3) + 0
    
    idxa=[0]*nextra
    for w1 in wta3:
        for i in range(nextra):
            if idxa[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    idxa[i]=allwords[w1]
                    break
    
    idxb=[0]*nextra
    for w1 in wtb3:
        for i in range(nextra):
            if idxb[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    idxb[i]=allwords[w1]
                    break
    
    ida=[0]*nall
    for w1 in wta:
        for i in range(nall):
            if ida[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    ida[i]=allwords[w1]
                    break
    idb=[0]*nall
    for w1 in wtb:
        for i in range(nall):
            if idb[i]==0:
                if istrain:
                    if w1 not in allwords:
                        allwords[w1] = len(allwords)+1
                if istrain or (w1 in allwords):
                    idb[i]=allwords[w1]
                    break
    
    df.loc[index, fldprefix + '_1'] = common0
    df.loc[index, fldprefix + '_2'] = common2
    df.loc[index, fldprefix + '_3'] = common3
    df.loc[index, fldprefix + '_4'] = common1
    df.loc[index, fldprefix + '_5'] = diff_whats
    df.loc[index, fldprefix + '_6'] = repeats1
    df.loc[index, fldprefix + '_7'] = diff_words2
    df.loc[index, fldprefix + '_8'] = synonyms1
    df.loc[index, fldprefix + '_9'] = diff_words3
    df.loc[index, fldprefix + '_10'] = diff_countries
    df.loc[index, fldprefix + '_11'] = nants
    df.loc[index, fldprefix + '_12'] = diff_whats+diff_countries+nants
    df.loc[index, fldprefix + '_13'] = nota+notb
    df.loc[index, fldprefix + '_14'] = nworda+nwordb
    df.loc[index, fldprefix + '_15'] = len(common_caps)
    df.loc[index, fldprefix + '_16'] = len(diff_caps1)
    df.loc[index, fldprefix + '_17'] = len(diff_caps2)
    df.loc[index, fldprefix + '_18'] = len(diff_caps1)+len(diff_caps2)
    col = 19
    for i in range(nfirst):
        df.loc[index, fldprefix + '_'+str(col)] = idfa[i]
        col+=1
        df.loc[index, fldprefix + '_'+str(col)] = idfb[i]
        col+=1
    for i in range(nlast):
        df.loc[index, fldprefix + '_'+str(col)] = idla[i]
        col+=1
        df.loc[index, fldprefix + '_'+str(col)] = idlb[i]
        col+=1
    for i in range(nall):
        df.loc[index, fldprefix + '_'+str(col)] = ida[i]
        col+=1
        df.loc[index, fldprefix + '_'+str(col)] = idb[i]
        col+=1
    for i in range(nextra):
        df.loc[index, fldprefix + '_'+str(col)] = idxa[i]
        col+=1
        df.loc[index, fldprefix + '_'+str(col)] = idxb[i]
        col+=1
        
    if k>=block:
        k=0
        print (index)


total_cols = 18+(nfirst+nlast+nall+nextra)*2

nrow = len(df)
for i in range(1,total_cols+1):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    df[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname+","+str(nrow))
