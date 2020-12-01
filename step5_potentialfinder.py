#!/usr/bin/env python
# coding: utf-8

#############
# @authors: Mathias Riechart, Roshan Bhandari, Abhijeet Aamle, Abhimanyu
# This code is used to pre process datasets and later feed into the functions.py that actually does the 
# exponential and logistic fit.
#############

from function import *


global_x=[]
from os import listdir
from os.path import isfile,isdir, join
import codecs
import cchardet
import random
from io import StringIO
import csv
#import potentialfinder
import pandas as pd


def get_encoding(file,skiprows=0):
    testStr = b''
    count = 0
    with open(file, 'rb') as x:
        line = x.readline()
        while line and count < 5000:  #Set based on lines you'd want to check
            for _ in range(0,skiprows): #read in only those rows which are later used for sampling as well 
                try:
                    next(x)
                except StopIteration:
                    pass
            testStr = testStr + line
            count = count + 1
            line = x.readline()
    result=cchardet.detect(testStr)
    if result['confidence'] is None:
        return ''
    if result['confidence']<1:
        print('# Encoding Detection: ' + result['encoding']+' (confidence:' + str(result['confidence'])+")")
    return result['encoding']

def buffered_row_count(file):
    count = 0
    thefile = open(file, 'rb')
    while 1:
        buffer = thefile.read(8192*1024)
        if not buffer: break
        count += buffer.count(b'\n')
    thefile.close(  )
    return count

def read_every_nth_row_to_byte(file,n_th):
    ## required as pandas.read_csv with the skip function takes too much memory. 
    ## --> first load every nth row as byte --> then load this into pandas
    with open(file, 'rb') as x:
        byteStr = b''
        line = x.readline()
        #line = line.replace(b'\n', b'')+b'\n'
        #print(line)
        while line:  #Set based on lines you'd want to check
            for _ in range(0,n_th): #read in only those rows which are later used for sampling as well 
                try:
                    next(x)
                except StopIteration:
                    pass
            byteStr = byteStr + line
            line = x.readline()
            #line = line.replace(b'\n', b'')+b'\n'
            #print(line)
        byteStr = byteStr + line
    return byteStr


def get_last_analyzed_file(analyzed_files):
    if isfile(analyzed_files):
        analyzed_files_obj = open(analyzed_files, "r")
        allfiles=analyzed_files_obj.read().splitlines() 
        analyzed_files_obj.close()
        
        return allfiles[-1]
    else:
        return ''

def update_analyzed_files(file_to_add,analyzed_files):
    analyzed_files_obj = open(analyzed_files, "a+")
    analyzed_files_obj.write(file_to_add+'\n')
    analyzed_files_obj.close()

n_rows=10000

path="/mnt/disks/localdisk/new_sampled_data1/"
debug=True
analyzed_files="analyzed_files.txt"
outputtablefile='outputtable_selected.csv'
skip_files=True
##read list of files that have already been analyzed (delete this file to force restart analysis on all files):
last_analyzed_file=get_last_analyzed_file(analyzed_files)
print("Last analyzed File: "+last_analyzed_file)

if last_analyzed_file=="":
    firstrow=True
else:
    firstrow=False
    resulttable=pd.read_csv("/mnt/disks/localdisk/code/outputs_selected3/table/"+outputtablefile,sep=";")
    
print("TEST:: ", listdir(path))

for username in listdir(path):
    print(username)
    if True:
    #if username in authorlist:
        if True:
            for datatablename in [username]:
                if True:
                #if datatablename in folderlist:
                    if True:
                        for filename in [join(path, username)]:
                            folder_and_file=join(path,username)
                            if isfile(folder_and_file):
                                if filename[-3:]!='csv':
                                    continue
                                outputTableName=path+'_'+username
                                if path+username == last_analyzed_file.replace('\\','') or last_analyzed_file=='':
                                    skip_files=False
                                if skip_files:
                                    print("#Skipping:"+ folder_and_file)
                                    continue
                                print("#Computing:"+ folder_and_file)
                                update_analyzed_files(folder_and_file,analyzed_files)
                                try:
                                    
                                    num_lines = buffered_row_count(folder_and_file) #requires no encoding 
                                    if num_lines<=n_rows:
                                        n_th=0
                                    else:
                                        n_th = round(num_lines/n_rows)  # every 100th line = 1% of the lines
                                    if num_lines == 0:
                                        print(folder_and_file + " seems empty.")
                                        continue
                                    encoding=get_encoding(folder_and_file,n_th)
                                    print("-> Encoding:"+encoding)
                                    try:
                                        if encoding=='':
                                            datastr=str(read_every_nth_row_to_byte(folder_and_file,n_th))
                                        else:
                                            datastr=str(read_every_nth_row_to_byte(folder_and_file,n_th),encoding=encoding)
                                    except:
                                        print("Skipped because of encoding identification error.")
                                        continue

                                    separators=[',',';','|','\t']
                                    quotechars=['"',"'"]
                                    worked=False
                                    for separator in separators:
                                        for quotechar in quotechars:
                                            if not worked:
                                                try:
                                                    data=StringIO(datastr)
                                                    print("trying " + separator + " | " + quotechar)
                                                    sqlDFTemp = pd.read_csv(
                                                        data,
                                                        sep = separator,
                                                        encoding=encoding,
                                                        quotechar=quotechar
                                                    )#, error_bad_lines=False)
                                                    if len(sqlDFTemp.columns)==1:
                                                        raise Exception(
                                                            '###separator has not worked out if there is only 1 column')
                                                    worked=True
                                                except:
                                                    try:
                                                        data=StringIO(datastr)
                                                        print("trying " + separator + " | " + quotechar)
                                                        sqlDFTemp = pd.read_csv(
                                                            data,
                                                            sep = None,
                                                            encoding=encoding,
                                                            engine = 'python',
                                                            error_bad_lines=False
                                                        )
                                                        worked=True
                                                    except:
                                                        pass
                                    if worked==False:
                                        print("skipped")
                                        continue
                                    
                                        #sqlDFTemp=pd.read_csv(join(path,username,datatablename,filename), sep = None, engine = 'python')
                                except UnicodeDecodeError:
                                    print ("Unicode Decode Error")
                                except:
                                    if debug:
                                        raise
                                    else:
                                        print('error parsing file: '+join(path,username))
                                        continue
                                
                                for fraction in [1,0.5,0.25]:
                                    temptable=find_exponents(sqlDFTemp,
                                                            fractionToAnalyze=fraction,
                                                            outputPath='outputs_selected3',
                                                            outputTable=True,
                                                            outputPlots=True,
                                                            outputTablename=outputTableName,
                                                            logToScreen=False,
                                                            columnFillThreshold=0.5,
                                                            exp_b_threshold=0.00000000002,
                                                            exp_r_s_threshold=0.7,
                                                            maxrows=n_rows,
                                                            debug=False)
                                    if firstrow:
                                        resulttable=temptable
                                        firstrow=False
                                    else:
                                        if not temptable is None:
                                            resulttable=resulttable.append(temptable,ignore_index=True)
                                            try:
                                                resulttable.to_csv("/mnt/disks/localdisk/code/outputs_selected3/table/"+outputtablefile,sep=";")
                                            except:
                                                print("Writing error. Resuming.")

resulttable.to_csv("/mnt/disks/localdisk/code/outputs_selected3/table/outputtable.csv",sep=";")

