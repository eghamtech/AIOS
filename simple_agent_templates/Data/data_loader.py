#start_of_parameters
#key=source_filename;  type=constant;  value=train_quora.csv
#end_of_parameters

# This script will scan your file for string columns and convert them to dictionaries.
# Note, that this operation can be done only once.
# Provide correct source filename in agent parameters
# workdir must be setup in 'Constants' area of AIOS
# target filename must be same as 

source_filename = "{source_filename}"

####################################################

newfilename = trainfile
import pandas as pd
df = pd.read_csv(workdir+source_filename)

char_cols = list(df.select_dtypes(include=['object']).columns)
print (char_cols)

def make_dict(col):
	a1 = col.unique()
	a1 = [x for x in a1 if str(x) != 'nan']
	keys1 = range(1, len(a1)+1)
	return dict(zip(a1, keys1))


for cname in char_cols:
	dict1 = make_dict(df[cname].fillna(''))
	df[cname] = df[cname].fillna('').map(dict1)
	pd.DataFrame(list(dict1.items()), columns=['value', 'key'])[['key','value']].to_csv(workdir+'dict_'+cname+'.csv')    #save new column dict

df.to_csv(workdir+newfilename, index=False)
	
for cname in df.columns:
	if cname in char_cols:
		is_dict="Y"
	else:
		is_dict="N"
	print ("#add_field:"+cname+","+is_dict+","+newfilename)
