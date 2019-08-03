import pandas as pd

def input_true_false(text):
    ina = input(text)
    try:
        ina = int(ina)
        if ina == 0:
            return False
        else:
            return True
    except:
        input_true_false(text)
    
file_name = '30-07-data.csv'
en_name = '30-07-data-en.csv'
vi_name = '30-07-data-vi.csv'
df = pd.read_csv(file_name,   encoding='utf-8')

allow_lang = ['en','vi']
default_lang = 'vi'
df_col = ["sentence","brand_name","info","index","content","total","thank_you"]

en_df = pd.DataFrame(columns=df_col)
vi_df = pd.DataFrame(columns=df_col)
for index, row in df.iterrows():
    print(row['sentence'])
    lang_en = input_true_false("Is en (0/1):")
    if lang_en:
        en_df.loc[en_df.shape[0]] = row
        en_df.to_csv(en_name, index = False,  encoding='utf-8')
    else:
        vi_df.loc[vi_df.shape[0]] = row
        vi_df.to_csv(vi_name, index = False,  encoding='utf-8')
