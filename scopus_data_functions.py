import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, math
from scipy import stats
import pycountry_convert as pc
from collections import Counter
import glob


'''
To put raw scopus output in the format needed for bbibli_functions.py
'''

def format_scopus(data_paths, output_directory):
    '''data_paths is list of paths to .csv files'''

    k=0
    for d in data_paths:
        print('Working on:', k, '/' , len(data_paths))

        data = pd.read_csv(d)
        data_cleaned0 = pd.DataFrame()

        for a in authors_random:
            a_papers = data[data['Author(s) ID'].str.contains(str(a))]
            a_papers['AU-ID'] = a
            a_papers['citedby-count'] = a_papers['Cited by']
            a_papers['authors'] = a_papers['Author(s) ID']
            a_papers = a_papers[['AU-ID', 'EID', 'Year', 'Document Type',
               'citedby-count', 'ISSN', 'authors',]] 
            data_cleaned0 = data_cleaned0.append(a_papers)

        data_cleaned0.to_csv(output_directory+str(k)+'.csv')
        k+=1