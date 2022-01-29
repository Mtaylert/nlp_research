import pandas as pd
KEEPS = ['item_id','query_mod','question','review','human_ans_spans','human_ans_indices']
store = []
for t in ['electronics','grocery']:
    df = pd.read_csv(f'{t}.csv')
    df = df[KEEPS]
    store.append(df)
storage_df = pd.concat(store)
train = storage_df[storage_df['human_ans_spans']!='ANSWERNOTFOUND']
test = storage_df[storage_df['human_ans_spans']=='ANSWERNOTFOUND']
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)