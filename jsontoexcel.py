
import pandas as pd
import json
##inputfile##
with open('file.json') as jsonfile:
    data = json.load(jsonfile)
Category = []
modules =[]
for key , value in data.items():
    Category.append(key)
    for k1, v1 in value.items():
        modules.append(k1)
print(Category)
print(modules)
modules = list(set(modules))
output = {'Sno':[], 'Category':[], 'Module': [], 'TestCase':[]}
count = 1
for i in Category:
    for j in modules:
        try:
            for k in data[i][j]['TestCases'][0]:
                output['Category'].append(i)
                output['Module'].append(j)
                output['TestCase'].append(k)
                output['Sno'].append(count)
                count+=1
        except:
            pass
##outputfile##
pd.DataFrame(output).to_excel('finalextract.xlsx',index=False)
