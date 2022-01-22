import pandas as pd

df = pd.read_csv('QnA.csv')
train = df[df['human_ans_spans']!='ANSWERNOTFOUND']
test = df[df['human_ans_spans']=='ANSWERNOTFOUND']
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)