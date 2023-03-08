import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

df_1 = pd.read_csv('Switches.csv')
df_2 = pd.read_csv('router.csv')
df_3 = pd.read_csv('machines.csv')
df_4 = pd.read_csv('gateway.csv')
df_5 = pd.read_csv('partreplace.csv')
df_6 = pd.read_csv('vietnam.csv')

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')

print('\n\n')
question = input("Enter Question: ")
text=""

#switch.csv
text=text+"There are "+str(df_1['Machine Type'].value_counts()['Smart Switch'])+" Smart Switch Machines."

text=text+" There are "+str(df_1['Machine Type'].value_counts()['Vision Quadra'])+" Vision Quadra Machines."

text = text+" There are " + str(df_1[df_1['Country Name']=='Japan']['Machine Type'].value_counts()["Smart Switch"])+" Smart Switch Machines in Japan."

text = text+" There are "+str(df_1[df_1['Delivered Site']=='Cisco']['Machine Type'].value_counts()["Smart Switch"])+" Smart Switches delivered to industry Cisco."

text = text+" There are "+str(df_1[df_1['Country Name']=='USA']['Machine Type'].value_counts()['Backend Switch'])+" Backend Switches delivered to USA"

text = text+" The price of Router B01 is " + str(df_1[df_1['Machine Model'] == 'B01']['Price'].values).replace('[]', '').replace(']', '')+"."


#router.csv
text = text+" The most sold router model in India was "+str(df_2['Machine Model'].mode().values)+"."

text = text+" Total number of routers sold in india are "+str(df_2[df_2['Country Name']=='India']['Machine Type'].value_counts()['Router'])+"."

text = text+" The prices of Router in India and USA are "+str(int(df_2[(df_2['Country Name'] == 'India') & (df_2['Machine Type'] == 'Router')]['Price'].mean()))+" and "+str(int(df_2[(df_2['Country Name'] == 'USA') & (df_2['Machine Type'] == 'Router')]['Price'].mean()))+"."


df_2['Delivered Date']=pd.to_datetime(df_2['Delivered Date'])

text = text+" Most sold machines in India last year is "+str(df_2[df_2['Delivered Date'].dt.year==2022]['Machine Model'].mode().values)+"."

text = text+" Most sold machines in India last three years is " + str(df_2[(df_2['Delivered Date'].dt.year == 2021) | (df_2['Delivered Date'].dt.year == 2022) | (df_2['Delivered Date'].dt.year == 2023)]['Machine Model'].mode().values)+"."

text = text+" Most sold machines in India this month is " + str(df_2[(df_2['Delivered Date'].dt.month == 3)]['Machine Model'].mode().values)+"."

text = text+" Most sold machines in India 2022 is " + str(df_2[(df_2['Delivered Date'].dt.year == 2022)]['Machine Model'].mode().values)+"."

text = text+" Most sold machines in Jan 2022 is " + str(df_2[(df_2['Delivered Date'].dt.year == 2022) & (df_2['Delivered Date'].dt.month == 1)]['Machine Model'].mode().values)+"."

#machines.csv
df_3['Fault Date'] = pd.to_datetime(df_3['Fault Date'])

text = text+" The latest fault events of machine code 555 is "+ str(df_3[df_3['Fault Type'] == 555].loc[0]['Machine Model'])+"."


if(((df_3['Fault Date'].dt.year == 2023) & (df_3['Fault Date'].dt.month == 3) & (df_3['Fault Date'].dt.day == 8)).count() > 0):
    text = text + " Yes, there is a fault for machine 567 today."
if ((df_3['Fault Type'] == 567).count() > 0):
    text = text+" Yes, there is part replaced for switch machine code 567."

text = text+" Number of routers serviced today are " + str(df_3[df_3['Fault Date'].dt.month == 3]['Machine Model'].count())+"."

text = text+" Number of routers serviced in last month are " + str(df_3[df_3['Fault Date'].dt.month == 2]['Machine Model'].count())+"."


#gateway.csv
df_4['Delivered Date'] = pd.to_datetime(df_4['Delivered Date'])

text = text+" The gateway machine models in United Kingdom are " + str(df_4[(df_4['Country Name'] == 'UK') & (df_4['Machine Type'] == 'Gateway')]['Machine Model'].values)+"."

text = text + " The price range of Gateway machines in UK are between " + str(df_4[df_4['Country Name'] == "UK"]['Price'].min())+" and " + str(df_4[df_4['Country Name'] == "UK"]['Price'].max())+"."

text=text+" The average price of gateway in UK is "+str(df_4[df_4['Country Name']=="UK"]['Price'].mean())+"."

text=text+" The number of gateways delivered last 5 years were "+str(df_4[df_4['Delivered Date'].dt.year>2018].count()['Machine Model'])+"."


#partreplace.csv
df_5['Date'] = pd.to_datetime(df_5['Date'])

text = text+" The part replace for the code 7890 is " + \
    str(df_5[df_5['Fault Type'] == '7890'].loc[0]['Part Name'])+"."

text = text+" The parts replaced for machine 7890 in last 5 years was " + str(df_5[df_5['Date'].dt.year > 2018]['Part Name'].values).replace('[', '').replace(']', '').replace("'",'')+"."

text=text+" The number of parts replaced for notification update this year was "+str(df_5[df_5['Date'].dt.year==2023].count()['Part Name'])+"."


#vietname.csv

df_6_1=df_6.drop_duplicates(subset=['Software Version'])


text=text+" The software version of switches in Vietnam are "+str(df_6_1['Software Version'].values).replace('[','').replace(']','').replace("'",'')+"."

text=text+" The number of machine delivered in vietnam is "+str(df_6['Machine Model'].count())+"."

text=text+" The number of VG40A machines in vietnam is "+str(df_6[df_6['Machine Type']=='VG40A']['Machine Model'].count())+"."



input_ids = tokenizer.encode(question, text)


tokens = tokenizer.convert_ids_to_tokens(input_ids)

sep_idx = input_ids.index(tokenizer.sep_token_id)
#number of tokens in segment A (question) - this will be one more than the sep_idx as the index in Python starts from 0
num_seg_a = sep_idx+1
#number of tokens in segment B (text)
num_seg_b = len(input_ids) - num_seg_a
#creating the segment ids
segment_ids = [0]*num_seg_a + [1]*num_seg_b
#making sure that every input token has a segment id
assert len(segment_ids) == len(input_ids)

#token input_ids to represent the input and token segment_ids to differentiate our segments - question and text
output = model(torch.tensor([input_ids]),
               token_type_ids=torch.tensor([segment_ids]))

answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)
if answer_end >= answer_start:
    answer = " ".join(tokens[answer_start:answer_end+1])
    print("\nQuestion:\n{}".format(question.capitalize()))
    print("\nAnswer:\n{}.".format(answer.capitalize()).replace('#','').replace('[','').replace(']',''))
else:
    print("I am unable to find the answer to this question. Can you please ask another question?")

