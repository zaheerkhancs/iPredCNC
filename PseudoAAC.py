#!/usr/bin/env python
# _*_coding:utf-8_*_
import datetime
import re,sys,os,platform
import math

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from util import *
import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np
import time

def readFASTA(fileName):
    with open(fileName , 'r') as file:
        v = []
        genome = ''
        for line in file:
            if line[0] != '>':
                genome += line.strip()
            else:
                v.append(genome)
                genome = ''
        v.append(genome)
        del v[0]
        return v


def Rvalue(aa1,aa2,AADict,Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PAAC(fastas,lambdaValue=30,w=0.05,**kw):
    if minSequenceLengthWithNormalAA(fastas) < lambdaValue + 1:
        print(
            'Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
        return 0
    dataFile = re.sub('codes$','',os.path.split(os.path.realpath(__file__))[
        0]) + r'\data\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$','',
                                                                              os.path.split(os.path.realpath(__file__))[
                                                                                  0]) + '/data/PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    a_a = ''.join(records[0].rstrip().split()[1:])
    a_a_dict = {}
    for i in range(len(a_a)):
        a_a_dict[a_a[i]] = i
    a_a_property = []
    a_a_property_names = []
    for i in range(1,len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        a_a_property.append([float(j) for j in array[1:]])
        a_a_property_names.append(array[0])

    AAProperty1 = []
    for i in a_a_property:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = ['#']
    for aa in a_a:
        header.append('bsp-' + aa)
    for n in range(1,lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
        # header.append('label')
    encodings.append(header)

    counter = 1
    for i in fastas:
        import time
        starttime = datetime.datetime.now()
        name,sequence = i[0],re.sub('-','',i[1])
        code = [name]
        theta = []
        for n in range(1,lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j],sequence[j + n],a_a_dict,AAProperty1) for j in range(len(sequence) - n)]) / (
                    len(sequence) - n))
        myDict = {}
        for aa in a_a:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in a_a]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        rfcode = re.sub(r'[^#0-9.,a-zA-Z]',"",str(code))
        encodings.append(rfcode)
        """ else class label append"""
        '''
        if counter < 179:
            rfcode.append(1)
            encodings.append(rfcode)
        else:
            rfcode.append(0)
            encodings.append(rfcode)
        counter = counter + 1
        end = datetime.datetime.now() - starttime
        print("Counter :" + str(counter) + "   Time Taken: " + str(end.microseconds))
        '''
    return encodings


if __name__ == '__main__':
    # fastas = readFasta('data/testpr.fasta')
    fastas = readFasta('data/datacnc.fasta')
    lambdaValue = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
    encodings = PAAC(fastas, lambdaValue)
    # result = re.sub(r'[^#0-9.,a-zA-Z]', "", str(encodings))
    with open('data/feature-gen/fpsc.csv','w+') as f:
        # with open('data/feature-gen/temptest.txt','w+') as f:
        for item in encodings:
            f.write("%s\n" % item)
        f.close()
