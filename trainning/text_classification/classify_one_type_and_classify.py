import pandas as pd
import numpy as np

# def get_col(info_column):
#     choose = len(info_column)

#     while choose > (len(info_column) - 1):
#         for icol, colname in enumerate(info_column):
#             print(icol,colname,end='\t')
#         print()
#         choose = int(input("Choose: "))
    
#     return choose

# origin_data = pd.read_csv('31-07-vigroupped.csv')

# filtered_data = origin_data.query('total>0')
# info_column = ['title','address','phone','datetime','table','client_no','cashier','website']
# total_column = ['total','pay','discount','text_money','other']
# new_pd = pd.DataFrame(filtered_data['sentence'],columns = ['sentence'])
# for col in info_column:
#     new_pd[col] = 0

# print(new_pd)


# for index, col in new_pd.iterrows():
#     print(col['sentence'])
#     coli = get_col(total_column)
        
#     new_pd.loc[index,total_column[coli]] = 1

#     new_pd.to_csv('total_df.csv',index=False)

# import math
# origin_data = pd.read_csv('total_df.csv')
# # print(math.isnan(origin_data['pay'][0]))
# for index, row in origin_data.iterrows():
#     for col in origin_data:
#         if not type(row[col]) == type(''):
#             if math.isnan(row[col]):
#                 origin_data.loc[index,col] = 0

# print(origin_data)
# origin_data.to_csv('total_df.csv',index=False)