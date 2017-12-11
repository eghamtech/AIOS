#start_of_parameters
#key=fieldname;  type=constant;  value=enter_fieldname
#key=filename;  type=constant;  value=enter_filename
#key=row_number;  type=constant;  value=enter_row_number
#end_of_parameters
import pandas as pd
df = pd.read_csv(workdir + "{filename}")
print (df.at[{row_number}, "{fieldname}"])