#no_permutation

if 'dicts' not in globals():
    dicts = {}

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.metrics import edit_distance

class cls_agent_{id}:
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd

    col_definition1 = "{random_dict}"
    col1 = col_definition1.split("|")[0]
    file1 = col_definition1.split("|")[1]
    col_definition2 = "{random_dict}"
    col2 = col_definition2.split("|")[0]
    file2 = col_definition2.split("|")[1]
    result_id = {id}
    field_prefix = 'nltk_'

    import string
    import nltk.corpus
    import nltk.stem.snowball
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    stopwords.append('')
    stopwords.append('would')
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    from nltk.corpus import wordnet
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    from nltk.tag.perceptron import PerceptronTagger
    tagger = PerceptronTagger()
    from collections import Counter
    import re, math
    import pycountry
    
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

    def func(self, s):
        return s[:1].lower() + s[1:] if s else ''
    
    def __init__(self):
        self.fldprefix = self.field_prefix + str(self.result_id)
        
        for c in list(self.pycountry.countries):
            cname = c.name.lower()
            self.cn.append(cname)
            self.cn.append(cname + 'n')
    
    def get_cosine(self, vec1, vec2):
        vec1 = self.Counter(vec1)
        vec2 = self.Counter(vec2)
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = self.math.sqrt(sum1) * self.math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def get_wordnet_pos(self, pos_tag):
        if pos_tag[1].startswith('J'):
            return (pos_tag[0], self.wordnet.ADJ)
        elif pos_tag[1].startswith('V'):
            return (pos_tag[0], self.wordnet.VERB)
        elif pos_tag[1].startswith('N'):
            return (pos_tag[0], self.wordnet.NOUN)
        elif pos_tag[1].startswith('R'):
            return (pos_tag[0], self.wordnet.ADV)
        else:
            return (pos_tag[0], self.wordnet.NOUN)

    def get_intersection(self, a1, a2):
        nom = len(set(a1).intersection(a2))
        denom = len(set(a1).union(a2))
        if len(a1)==0 and len(a2)==0:
            return 1.0
        return 1.0*nom/(denom+0.0001)


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
        
            common1 = len(set(sline1.lower().split()).intersection(set(sline2.lower().split())))

            wta = word_tokenize(sline1.lower())
            wtb = word_tokenize(sline2.lower())

            tagset = None
            pos_a = map(self.get_wordnet_pos, self.nltk.tag._pos_tag(wta, tagset, self.tagger))
            pos_b = map(self.get_wordnet_pos, self.nltk.tag._pos_tag(wtb, tagset, self.tagger))
            tokens_a = [token.lower().strip(self.string.punctuation) for token in wta \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_b = [token.lower().strip(self.string.punctuation) for token in wtb \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_adj_a = [token.lower().strip(self.string.punctuation) for token, pos in pos_a \
                            if pos == self.wordnet.ADJ and token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_adj_b = [token.lower().strip(self.string.punctuation) for token, pos in pos_b \
                            if pos == self.wordnet.ADJ and token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_verb_a = [token.lower().strip(self.string.punctuation) for token, pos in pos_a \
                            if pos == self.wordnet.VERB and token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_verb_b = [token.lower().strip(self.string.punctuation) for token, pos in pos_b \
                            if pos == self.wordnet.VERB and token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_noun_a = [token.lower().strip(self.string.punctuation) for token, pos in pos_a \
                            if pos == self.wordnet.NOUN and token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_noun_b = [token.lower().strip(self.string.punctuation) for token, pos in pos_b \
                            if pos == self.wordnet.NOUN and token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_adv_a = [token.lower().strip(self.string.punctuation) for token, pos in pos_a \
                            if pos == self.wordnet.ADV and token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_adv_b = [token.lower().strip(self.string.punctuation) for token, pos in pos_b \
                            if pos == self.wordnet.ADV and token.lower().strip(self.string.punctuation) not in self.stopwords]

            res1 = self.get_intersection(tokens_a, tokens_b)
            res2 = self.get_intersection(tokens_adj_a, tokens_adj_b)
            res3 = self.get_intersection(tokens_verb_a, tokens_verb_b)
            res4 = self.get_intersection(tokens_noun_a, tokens_noun_b)
            res5 = self.get_intersection(tokens_adv_a, tokens_adv_b)

            stems_a = [self.stemmer.stem(token) for token in tokens_a]
            stems_b = [self.stemmer.stem(token) for token in tokens_b]
            stems_adj_a = [self.stemmer.stem(token) for token in tokens_adj_a]
            stems_adj_b = [self.stemmer.stem(token) for token in tokens_adj_b]
            stems_verb_a = [self.stemmer.stem(token) for token in tokens_verb_a]
            stems_verb_b = [self.stemmer.stem(token) for token in tokens_verb_b]
            stems_noun_a = [self.stemmer.stem(token) for token in tokens_noun_a]
            stems_noun_b = [self.stemmer.stem(token) for token in tokens_noun_b]
            stems_adv_a = [self.stemmer.stem(token) for token in tokens_adv_a]
            stems_adv_b = [self.stemmer.stem(token) for token in tokens_adv_b]

            res6 = self.get_intersection(stems_a, stems_b)
            res7 = self.get_intersection(stems_adj_a, stems_adj_b)
            res8 = self.get_intersection(stems_verb_a, stems_verb_b)
            res9 = self.get_intersection(stems_noun_a, stems_noun_b)
            res10 = self.get_intersection(stems_adv_a, stems_adv_b)

            lemma_a = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_a \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_b = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_b \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_adj_a = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_a \
                        if pos == self.wordnet.ADJ and token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_adj_b = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_b \
                        if pos == self.wordnet.ADJ and token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_verb_a = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_a \
                        if pos == self.wordnet.VERB and token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_verb_b = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_b \
                        if pos == self.wordnet.VERB and token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_noun_a = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_a \
                        if pos == self.wordnet.NOUN and token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_noun_b = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_b \
                        if pos == self.wordnet.NOUN and token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_adv_a = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_a \
                        if pos == self.wordnet.ADV and token.lower().strip(self.string.punctuation) not in self.stopwords]
            lemma_adv_b = [self.lemmatizer.lemmatize(token.lower().strip(self.string.punctuation), pos) for token, pos in pos_b \
                        if pos == self.wordnet.ADV and token.lower().strip(self.string.punctuation) not in self.stopwords]

            res11 = self.get_intersection(lemma_a, lemma_b)
            res12 = self.get_intersection(lemma_adj_a, lemma_adj_b)
            res13 = self.get_intersection(lemma_verb_a, lemma_verb_b)
            res14 = self.get_intersection(lemma_noun_a, lemma_noun_b)
            res15 = self.get_intersection(lemma_adv_a, lemma_adv_b)

            sum1=0.0
            sum2=0.0
            for j in range(0, len(tokens_a)):
                try:
                    sum1+=float(tokens_a[j])
                except:
                    pass
            for j in range(0, len(tokens_b)):
                try:
                    sum2+=float(tokens_b[j])
                except:
                    pass

            sline_tok_a = ' '.join(tokens_a)
            sline_tok_b = ' '.join(tokens_b)
            sline_stem_a = ' '.join(stems_a)
            sline_stem_b = ' '.join(stems_b)
            sline_lemm_a = ' '.join(lemma_a)
            sline_lemm_b = ' '.join(lemma_b)

            wtpa = [ " ".join(pair) for pair in self.nltk.bigrams(wta)]
            wtpb = [ " ".join(pair) for pair in self.nltk.bigrams(wtb)]

            n_what = int(("what" in wta)!=("what" in wtb))
            n_which = int(("which" in wta)!=("which" in wtb))
            n_how_much = int((("how much" in wtpa) or ("how many" in wtpa))!=(("how much" in wtpb) or ("how many" in wtpb)))
            n_why = int(("why" in wta)!=("why" in wtb))
            n_how = int(("how" in wta)!=("how" in wtb))
            n_when = int(("when" in wta)!=("when" in wtb))
            n_who = int(("who" in wta)!=("who" in wtb))

            n_total = 5*n_why + 3*n_when + 8*n_how_much + 3*n_how + 2*n_which + 2*n_what + 4*n_who

            n_both_what = int(("what" in wta) and ("what" in wtb))
            n_both_which = int(("which" in wta) and ("which" in wtb))
            n_both_how_much = int((("how much" in wtpa) or ("how many" in wtpa)) and (("how much" in wtpb) or ("how many" in wtpb)))
            n_both_why = int(("why" in wta) and ("why" in wtb))
            n_both_how = int(("how" in wta) and ("how" in wtb))
            n_both_when = int(("when" in wta) and ("when" in wtb))
            n_both_who = int(("who" in wta) and ("who" in wtb))


            syns_a = []
            for s1 in lemma_a:
                for k,j in enumerate(self.wordnet.synsets(s1)):
                    syns_a =syns_a + [j.lemma_names()[0]]
                    break

            syns_b = []
            for s1 in lemma_b:
                for k,j in enumerate(self.wordnet.synsets(s1)):
                    syns_b =syns_b + [j.lemma_names()[0]]
                    break

            res52 = self.get_intersection(syns_a, syns_b)

            wta0 = word_tokenize(sline1)
            wtb0 = word_tokenize(sline2)

            tokens_a0 = [token.strip(self.string.punctuation) for token in wta0 \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]
            tokens_b0 = [token.strip(self.string.punctuation) for token in wtb0 \
                        if token.lower().strip(self.string.punctuation) not in self.stopwords]

            lows_a = []
            caps_a = []
            lows_b = []
            caps_b = []
            for t in tokens_a0:
                if (t[0].lower()!=t[0]) or (t.lower() in self.cn):
                    caps_a.append(t.lower())

            for t in tokens_b0:
                if (t[0].lower()!=t[0]) or (t.lower() in self.cn):
                    caps_b.append(t.lower())

            for t in tokens_a:
                if (t not in caps_a) and (t in caps_b):
                    caps_a.append(t)
                elif t not in caps_a:
                    lows_a.append(t)

            for t in tokens_b:
                if (t not in caps_b) and (t in caps_a):
                    caps_b.append(t)
                elif t not in caps_b:
                    lows_b.append(t)

            res53 = self.get_intersection(lows_a, lows_b)
            res54 = self.get_intersection(caps_a, caps_b)



            df_run.set_value(index, self.fldprefix + '_1', res1)
            df_run.set_value(index, self.fldprefix + '_2', res2)
            df_run.set_value(index, self.fldprefix + '_3', res3)
            df_run.set_value(index, self.fldprefix + '_4', res4)
            df_run.set_value(index, self.fldprefix + '_5', res5)
            df_run.set_value(index, self.fldprefix + '_6', res6)
            df_run.set_value(index, self.fldprefix + '_7', res7)
            df_run.set_value(index, self.fldprefix + '_8', res8)
            df_run.set_value(index, self.fldprefix + '_9', res9)
            df_run.set_value(index, self.fldprefix + '_10', res10)
            df_run.set_value(index, self.fldprefix + '_11', res11)
            df_run.set_value(index, self.fldprefix + '_12', res12)
            df_run.set_value(index, self.fldprefix + '_13', res13)
            df_run.set_value(index, self.fldprefix + '_14', res14)
            df_run.set_value(index, self.fldprefix + '_15', res15)
            df_run.set_value(index, self.fldprefix + '_16', abs(len(sline1)-len(sline2)))
            df_run.set_value(index, self.fldprefix + '_17', abs(sum1-sum2))
            df_run.set_value(index, self.fldprefix + '_18', edit_distance(sline_tok_a, sline_tok_b))
            df_run.set_value(index, self.fldprefix + '_19', edit_distance(sline_stem_a, sline_stem_b))
            df_run.set_value(index, self.fldprefix + '_20', edit_distance(sline_lemm_a, sline_lemm_b))
            df_run.set_value(index, self.fldprefix + '_21', n_what)
            df_run.set_value(index, self.fldprefix + '_22', n_which)
            df_run.set_value(index, self.fldprefix + '_23', n_how_much)
            df_run.set_value(index, self.fldprefix + '_24', n_why)
            df_run.set_value(index, self.fldprefix + '_25', n_how)
            df_run.set_value(index, self.fldprefix + '_26', n_when)
            df_run.set_value(index, self.fldprefix + '_27', n_who)
            df_run.set_value(index, self.fldprefix + '_28', n_total)
            df_run.set_value(index, self.fldprefix + '_29', n_both_what)
            df_run.set_value(index, self.fldprefix + '_30', n_both_which)
            df_run.set_value(index, self.fldprefix + '_31', n_both_how_much)
            df_run.set_value(index, self.fldprefix + '_32', n_both_why)
            df_run.set_value(index, self.fldprefix + '_33', n_both_how)
            df_run.set_value(index, self.fldprefix + '_34', n_both_when)
            df_run.set_value(index, self.fldprefix + '_35', n_both_who)
            df_run.set_value(index, self.fldprefix + '_36', res52)
            df_run.set_value(index, self.fldprefix + '_37', res53)
            df_run.set_value(index, self.fldprefix + '_38', res54)
            df_run.set_value(index, self.fldprefix + '_39', len(sline1))
            df_run.set_value(index, self.fldprefix + '_40', len(sline2))
            df_run.set_value(index, self.fldprefix + '_41', len(sline1)-len(sline2))
            df_run.set_value(index, self.fldprefix + '_42', len(''.join(set(sline1.replace(' ', '')))))
            df_run.set_value(index, self.fldprefix + '_43', len(''.join(set(sline2.replace(' ', '')))))
            df_run.set_value(index, self.fldprefix + '_44', len(sline1.split()))
            df_run.set_value(index, self.fldprefix + '_45', len(sline2.split()))
            df_run.set_value(index, self.fldprefix + '_46', common1)
            df_run.set_value(index, self.fldprefix + '_47', self.get_cosine(sline_tok_a, sline_tok_b))
            df_run.set_value(index, self.fldprefix + '_48', self.get_cosine(sline_stem_a, sline_stem_b))
            df_run.set_value(index, self.fldprefix + '_49', self.get_cosine(sline_lemm_a, sline_lemm_b))
            df_run.set_value(index, self.fldprefix + '_50', len(set(tokens_noun_a).intersection(tokens_noun_b)))

            if i>=block and block>=1000:
                i=0
                print (index)

        df_run[self.fldprefix + '_17']=df_run[self.fldprefix + '_17'].fillna(value=0)

    def run(self, mode):
        print ("enter run mode " + str(mode))
        self.df = self.pd.read_csv(workdir+self.file1)[[self.col1]]
        self.df = self.df.merge(self.pd.read_csv(workdir+self.file2)[[self.col2]], left_index=True, right_index=True)
        
        self.run_on(self.df)
        
        nrow = len(self.df)

        for i in range(1,51):
            fld = self.fldprefix + '_' + str(i)
            fname = fld + '.csv'
            self.df[[fld]].to_csv(workdir+fname)
            print ("#add_field:"+fld+",N,"+fname+","+str(nrow))

    def apply(self, df_add):
        self.run_on(df_add)

agent_{id} = cls_agent_{id}()
