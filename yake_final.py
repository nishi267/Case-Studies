from __future__ import unicode_literals, print_function
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
import random
from  pathlib import Path
import pickle
from spacy.util import minibatch, compounding
import os
from os import path, mkdir
import spacy
import time
import numpy as np  
from string import digits
import json 
import yake
import pandas as pd 


#####################################################
# function 1- Text Preprocessing

def preprocess(text):
    text=text.replace('\n', ' ')
    text=text.replace('\t', ' ')
    text=text.lower()
    remove_dig=str.maketrans('','', digits)
    text=text.translate(remove_dig)
    return text
###################################################

# Function - Keywords Extractor from Requirements

def key_extract(level, reqlist):
    language = "en"
    max_ngram_size = 2
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    if level==1:
        numOfKeywords_1 = 2
        numOfKeywords_2 = 3
    else:
         numOfKeywords_1 = 5
         numOfKeywords_2 = 8
    masterlist1=[]
    masterlist3=[]
    for i in reqlist:
        cnt=0
        if(cnt<2):
            if(cnt==0):
                print(i[0])
                text=preprocess(i[0])
                custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords_1, features=None)
                keywords = custom_kw_extractor.extract_keywords(text)

                list1=[]
                for kw in keywords:
                    list1.append(kw)
                list1=sorted(list1, key=lambda x: x[1])
                temp1=[z[0] for z in list1 if z[1]]
                masterlist1.append(temp1)
                cnt=cnt+1

            if(cnt==1):
                print(i[1])
                text=preprocess(i[1])
                custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords_2, features=None)
                keywords = custom_kw_extractor.extract_keywords(text)
                list2=[]
                for kw in keywords:
                    list2.append(kw)
                list2=sorted(list2, key=lambda x: x[1])
                temp1=[z[0] for z in list2 if z[1]]
                masterlist3.append(temp1)
                cnt=cnt+1
    masterlist12=[x[0] for x in reqlist]
    masterlist=pd.DataFrame({"Req_Name": masterlist12, "Title_keys": masterlist1, "Desc_keys":masterlist3})
    masterlist.to_excel("keylist.xlsx")
    return masterlist
#################################################

    ## Clean and aggregate  Test Cases sheet

def cleanTestcases(df):
    df["Description_Design_Steps"]=df.Description_Design_Steps.apply(lambda x:str.lower(str(x)))
    df.drop(df[df["Description_Design_Steps"].str.contains("pre-requisite|prerequisite|pre-condition|precondition",
    na=False)].index, inplace=True)
    df=df.groupby(['Test_Name'],as_index=False).agg(lambda x: list(x))
    df["Description_Design_Steps1"]=df["Description_Design_Steps"]
    df["Description_Design_Steps"]=df["Description_Design_Steps"].apply(lambda x: ','.join(map(str,x)))
    df["Description_Design_Steps"]=df["Description_Design_Steps"].str.replace('[', ' ')
    df["Description_Design_Steps"]=df["Description_Design_Steps"].str.replace(']', ' ')
    df["Description_Design_Steps"]=df["Description_Design_Steps"].str.replace('\n', ' ')
    df["TCNameandDesc"]=df['Test_Name'] + df["Description_Design_Steps"]
    print(df.columns)
    df.to_excel("Agg.xlsx")
    return df

#######################################################
#poolcompare to create filter exclude list

def poolcomparefilter(df):
    import re
    df=cleanTestcases(df) 
    str1=""
    for i in range(len(df)):
        stre=df.iloc[i,3]
        str1=str1+stre
    str1=re.sub(r'[^\w\s]','',str1)
    lamba1=str1.split(" ")
    stop_words=set(stopwords.words('english'))
    lamba=[w for w in lamba1 if not w in stop_words]
    fdist1=nltk.FreqDist(lamba)
    import operator
    fdist1={r: fdist1[r] for r in sorted(fdist1, key=fdist1.get, reverse=True)}
    import itertools
    #out=dict(itertools.islice(fdist1.items(), 30))
   # words=list(out.keys())
    common =['version', "system", "data", "upgrade","options","columns","column" "type" , "displayed","variable" , "environment", "parameter", "parameters","outlook"]
    #words=words +common
    words=common
    return words
#####################################################

#clean postag
def postagclean(word):
    tokens=[word]
    d=nltk.pos_tag(tokens)
    r=d[0][1]
    return r

#Clean tag for browser

def tagother(req):
    tag_dict=[]
    nlp_model=spacy.load("trial")
    doc=nlp_model(req)
    for ent in doc.ents:
        print(f'{ent.label_.upper():{90}}-{ent.text}')
        if(ent.label_.upper() in ["B-BROWSER", "I-BROWSER"]):
            tag_dict.append(ent.text)
    return tag_dict




########################################################

##### Test Case Compare with Req Keywords######

def keysearchTC(masterlist, df):
    poolfilter=poolcomparefilter(df)
    keywords1=masterlist
    keywords1.index=keywords1.index.map(str)
    keywords1.to_excel("result/allkeys.xls")
    cnt=1
    req_diction={}
    for row in range(0,len(keywords1)):
        cnt1=1
        cnt2=1
        req_diction[keywords1.iloc[row,0]]={}
        req_diction[keywords1.iloc[row,0]]["TestCases"]=[]
        print(keywords1.iloc[row,0])
        listkeys_title=keywords1.iloc[row,1]
        listkeys_desc=keywords1.iloc[row,2]
        listkeys1=listkeys_title + listkeys_desc
        listkeys=[]
        for x in listkeys1:
            if x not in poolfilter:

                if len(x.split())==1:
                    d=postagclean(x)
                    if (d in ["NN", "NNS", "NNP", "NNPS"]):
                        listkeys.append(x)
                else:
                    listkeys.append(x)
        for x in listkeys:
            if(x not in poolfilter):
                print("Searching for keyword: ", x)
                frame1=df["TCNameandDesc"].str.contains(x, na=False)
                frame1=df[frame1]
                isempty=frame1.empty
                if(isempty==False):
                    if(cnt==1):
                        frame3=frame1
                        cnt=cnt+1
                    else:
                        frame3=pd.concat([frame3, frame1], ignore_index=True)
                        cnt=cnt+1
                    if(cnt1==1):
                        frame2=frame1
                        cnt1=cnt1+1
                    else:
                        frame2=pd.concat([frame2,frame1], ignore_index=True)
                        cnt1=cnt1+1
                else:
                    if(cnt1==1):
                        frame2=frame1 
            if(cnt2==1):
                cnt2=cnt2+1
                text=keywords1.iloc[row,0]
                browsertag=tagother(text)
                if(len(browsertag)>0):
                    searchfor=['ms edge', 'chrome','google chrome','firefox', 'safari','internet explorer']
                    frame1=df["TCNameandDesc"].str.contains('|'.join(searchfor))
                    frame1=df[frame1]
                    ######Extra
                   # frame1["Test_Name"]=frame1["Test_Name"].apply(lambda x: x +"--Browser")
                    isempty=frame1.empty
                    if(isempty==False):
                        if(cnt1==1):
                            frame2=frame1
                            cnt1=cnt1+1

                        else:
                            frame2=pd.concat([frame2,frame1], ignore_index=True)
                            cnt1=cnt1+1
        frame2=frame2.loc[frame2.astype(str).drop_duplicates().index]
        req_diction[keywords1.iloc[row,0]]['TestCases'].append(frame2['Test_Name'].values.tolist())
    return req_diction

#####   JSON to EXCEL

def jsontoexcel(filename):
    with open(filename) as jsonfile:
        data = json.load(jsonfile)
    Category = []
    modules =[]
    for key , value in data.items():
        Category.append(key)
        for k1, v1 in value.items():
            modules.append(k1)
    modules = list(set(modules))
    output = {'Sno':[], 'Category':[], 'Requirement': [], 'TestCase':[]}
    count = 1
    for i in Category:
        for j in modules:
            try:
                for k in data[i][j]['TestCases'][0]:
                    output['Category'].append(i)
                    output['Requirement'].append(j)
                    output['TestCase'].append(k)
                    output['Sno'].append(count)
                    count+=1
            except:
                pass
    pd.DataFrame(output).to_excel('result/finalextract1.xlsx',index=False)
    print("Process Successful.....File created!!!!")

###-----------------Execution Block --------------------------------------####

## Read Requirement from Excel
dfreq=pd.read_excel(r"D:/code-replicate/Requirements1.xlsx")
df_requirements=dfreq.drop(['Module name'], axis=1)
reqlist=df_requirements.values.tolist()

## Extracting Keywords from Requirement
masterlistlevel1=key_extract(1, reqlist)
masterlistlevel2=key_extract(2, reqlist)

## Reading Test Cases


df=pd.read_excel(r"D:/code-replicate/Testcasesfordefects_1.xlsx")
df=cleanTestcases(df)
### Searching for Keywords inside Test Cases
req_diction1=keysearchTC(masterlistlevel1,df)
req_diction2=keysearchTC(masterlistlevel2,df)
master_diction1={}
master_diction1["High Level Regression"]=req_diction1
master_diction1["Detailed Regression"]=req_diction2
with open('file1.json', 'w') as file:
    json.dump(master_diction1, file)
filename='file1.json'

## Converting JSON to EXCEL (count field is not captured)
jsontoexcel(filename)

for x, y in req_diction1.items():
    for k, dk in y.items():
        if k=="TestCases":
            for i in dk:
                count=len(i)
                dk1=['\n'.join([str(x) for x in i])]
                #dk1=['\n'.join([str(x) for x in i if x.find("--Browser") == -1])]
                req_diction1[x][k]=dk1
    req_diction1[x]['count']=count

for x,y in req_diction2.items():
    for k, dk in y.items():
        for i in dk:
            count=len(i)
            dk1=['\n'.join([str(x) for x in i])]
            req_diction2[x][k]=dk1
    req_diction2[x]['count']=count

    master_diction={}
    master_diction["High Level Regression"]=req_diction1
    master_diction["Detailed Regression"]=req_diction2

#### Storing final output as json file to be returned to FE

















