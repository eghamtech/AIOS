import pandas as pd
import numpy as np
import math
import re
import pycountry
cn = []
for c in list(pycountry.countries):
    cname = c.name.lower()
    cn.append(cname)
    cn.append(cname + 'n')        

cn+=['vietnam','laos','america','american','americans','oceania','banglore','chandigarh','pune','french','finnish','israil',
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

col_definition1 = "{random_dict}"
col1 = col_definition1.split("|")[0]
file1 = col_definition1.split("|")[1]
col_definition2 = "{random_dict}"
col2 = col_definition2.split("|")[0]
file2 = col_definition2.split("|")[1]

result_id = {id}

field_prefix = 'diffs_'

df = pd.read_csv(workdir+file1)[[col1]]
df = df.merge(pd.read_csv(workdir+file2)[[col2]], left_index=True, right_index=True)

dict1 = pd.read_csv(workdir+'dict_'+col1+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()
dict2 = pd.read_csv(workdir+'dict_'+col2+'.csv', dtype={'value': object}).set_index('key')["value"].to_dict()

df[col1] = df[col1].map(dict1)
df[col2] = df[col2].map(dict2)



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

def replacements(sline):
    res = sline
    for a in mapping_case:
        res = res.replace(a[0], a[1])
    res = res.lower()
    for a in mapping:
        res = res.replace(a[0], a[1])
    return res

def replacements_word(aword):
    res = list(aword)
    for i in range(len(res)):
        head, middle, tail = res[i].rpartition('ies')
        if len(tail)==0:
            res[i] = head+'y'
        for a in mapping_word:
            if res[i]==a[0]:
                res[i] = a[1]
    return res

def get_diffs2(sline1, sline2):
    wta = replacements_word(re.findall(r"[\w']+", sline1.lower()))
    wtb = replacements_word(re.findall(r"[\w']+", sline2.lower()))

    swta = set(wta)
    swtb = set(wtb)
    wta2 = list(swta-swtb)
    wtb2 = list(swtb-swta)
    
    wta3 = list(set(wta2)-sw)
    wtb3 = list(set(wtb2)-sw)
    
    cna = set(wta3).intersection(cn)
    cnb = set(wtb3).intersection(cn)
    
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
        sum1+=int(1000000.0/word_counts[wta3[i]])
        try:
            numbers1+=float(wta3[i])
        except:
            pass
    for i in range(len(wtb3)):
        sum2+=int(1000000.0/word_counts[wtb3[i]])
        try:
            numbers2+=float(wtb3[i])
        except:
            pass
                
    return common1, common2, wta3, wtb3, list(cna), list(cnb), sum1, sum2, numbers1, numbers2



print ( "calculating word counts" )
word_counts = {}

block = len(df)/50
k=block-1
for index, row in df.iterrows():
    k+=1
    if type(row[col1])==str:
        sline1 = replacements(row[col1])
    else:
        sline1 = ''
    if type(row['question2'])==str:
        sline2 = replacements(row[col2])
    else:
        sline2 = ''

    wta = replacements_word(re.findall(r"[\w']+", sline1))
    wtb = replacements_word(re.findall(r"[\w']+", sline2))

    for w in wta+wtb:
        if w in word_counts:
            word_counts[w]+=1
        else:
            word_counts[w]=1
    if k>=block:
        print (index)
        k=0

print ( "ok" )


func = lambda s: s[:1].lower() + s[1:] if s else ''


fldprefix = field_prefix + str(result_id)

block = int(len(df)/50)
i = block-1

for index, row in df.iterrows():
    i+=1
    if type(row[col1])==str:
        sline1 = func(row[col1])
    else:
        sline1 = ''
    if type(row[col2])==str:
        sline2 = func(row[col2])
    else:
        sline2 = ''

    
    ncomm1, ncomm2, a1, a2, cna, cnb, sum1, sum2, n1, n2 = get_diffs2(replacements(sline1), replacements(sline2))

    df.set_value(index, fldprefix + '_1', ncomm1) # count of common words
    df.set_value(index, fldprefix + '_2', len(a1)) # count of differ words in sentance 1
    df.set_value(index, fldprefix + '_3', len(a2)) # count of differ words in sentance 2
    df.set_value(index, fldprefix + '_4', len(a1)+len(a2)) # count of differ words in sentance 2
    df.set_value(index, fldprefix + '_5', len(cna)) # count of differ countries in sentance 1
    df.set_value(index, fldprefix + '_6', len(cnb)) # count of differ countries in sentance 2
    df.set_value(index, fldprefix + '_7', len(cna)+len(cnb))
    df.set_value(index, fldprefix + '_8', sum1) # total differ words importance of sentance 1
    df.set_value(index, fldprefix + '_9', sum2) # total differ words importance of sentance 2
    df.set_value(index, fldprefix + '_10', abs(sum1-sum2))
    df.set_value(index, fldprefix + '_11', n1) # sum of numbers in sentance 1
    df.set_value(index, fldprefix + '_12', n2) # sum of numbers in sentance 2
    df.set_value(index, fldprefix + '_13', abs(n1-n2))

    
    if i>=block:
        i=0
        print (index)


df[[fldprefix + '_11',fldprefix + '_12',fldprefix + '_13']]=df[[fldprefix + '_11',fldprefix + '_12',fldprefix + '_13']].fillna(value=0)

nrow = len(df)

for i in range(1,14):
    fld = fldprefix + '_' + str(i)
    fname = fld + '.csv'
    df[[fld]].to_csv(workdir+fname)
    print ("#add_field:"+fld+",N,"+fname+","+str(nrow))
