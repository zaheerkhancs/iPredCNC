import re
import datetime
import re,sys,os
from collections import Counter

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from util import *
import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np
import time

# !/usr/bin/env python
# _*_coding:utf-8_*_

import re


def Count(seq1,seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum


def stripchar(code):
    rfcode = re.sub(r'[^#0-9.,a-zA-Z]', "", str(code))
    rfcode =list(rfcode.split(','))
    return rfcode

def CTDC(fastas,**kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101','hydrophobicity_ARGP820101','hydrophobicity_ZIMJ680101','hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101','hydrophobicity_ENGD860101','hydrophobicity_FASG890101','normwaalsvolume',
        'polarity','polarizability','charge','secondarystruct','solventaccess')

    encodings = []
    header = ['#']
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append(p + '.G' + str(g))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], re.sub('-','',i[1])
        code = [name]
        for p in property:
            c1 = Count(group1[p],sequence) / len(sequence)
            c2 = Count(group2[p],sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
            rfcode= re.sub(r'[^#0-9.,a-zA-Z]',"",str(code))
        encodings.append(rfcode)
    return encodings

#def getctdcvect(filefasta):
if __name__ == '__main__':
    fastas = readFasta('data/datacnc.fasta')
    encodings = CTDC(fastas)
    # result = re.sub(r'[^#0-9.,a-zA-Z]', "", str(encodings))
    #with open('data/feature-gen/ftesting/fctdvect.csv', 'w+') as f:
    with open('data/feature-gen/fctd.csv','w+') as f:
        for item in encodings:
            f.write("%s\n" % item)
        f.close()
