import numpy as np
import pandas as pa
import csv
import re


def preprocess(csvfile):
    data = []
    remove_hashtags = re.compile(r'#\w+\s?')
    remove_friendtag = re.compile(r'@\w+\s?')
    remove_sarcasm = re.compile(re.escape('sarcasm'),re.IGNORECASE)
    remove_sarcastic = re.compile(re.escape('sarcastic'),re.IGNORECASE)
    line2 = ""
    for i in range(0,len(csvfile)):
        line = csvfile[i]
        if len(line[0:]) == 0:
            continue
        if len(line[0:])==1:
            line1 = line[0:][0]
            line1 = line1.replace('"','')
            if line1 != "/":
                line2 = line2 + " "+line1
            elif line1 == "/":
                    temp=line2
                    line2 = ""
                    if len(temp) == 0:
                        continue;
                    if(temp.count("/") > 0):
                        continue;
                    temp=remove_hashtags.sub('',temp)
                    if len(temp)>0 and 'http' not in temp and temp[0]!='@' and '\u' not in temp: 
                        temp=remove_friendtag.sub('',temp)
                        temp=remove_sarcasm.sub('',temp)
                        temp=remove_sarcastic.sub('',temp)
                        temp=' '.join(temp.split())
                        if len(temp.split())>2:
                            data.append(temp)
    data=list(set(data))
    data = np.array(data)
    return data

def main():
	csvfile = list(csv.reader(open('sarcasmfull.csv', 'rU'),delimiter='\n'))
	sarcasm_data = preprocess(csvfile)
	np.save('sarcasm_array',sarcasm_data)
	csvfile = list(csv.reader(open('nonsarcasmfull.csv', 'rU'),delimiter='\n'))
	non_sarcasm_data = preprocess(csvfile)
	np.save('non_sarcasm_array',non_sarcasm_data)
	print ("Saved")
	
main()
