import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, math
from scipy import stats
import pycountry_convert as pc
from collections import Counter

def n_authors(data):
    nauthors = [len(a.split(';')) for a in data['authors']]
    fig = plt.figure(facecolor='w')
    plt.hist(nauthors, bins=100, histtype='step');
    plt.xlabel('n authors');
    print('Averange Number of authors per paper: ', round(np.mean(nauthors), 1))
    print('Standard deviation:', round(np.std(nauthors), 2))

def n_papers_per_author(data):
    author_papers_count = np.asarray(list(Counter(data['AU-ID']).values()))
    fig = plt.figure(facecolor='w')
    plt.hist(author_papers_count, bins=int(len(data)/1000), histtype='step');
    plt.xlabel('n papers');
    print('Averange Number of papers per author: ', round(np.mean(author_papers_count), 1))
    print('Standard deviation:', round(np.std(author_papers_count), 2))
    
def h_index_expert(citations):
    
    citations = np.array(citations)
    n         = citations.shape[0]
    array     = np.arange(1, n+1)
    
    # reverse sorting
    citations = np.sort(citations)[::-1]
    
    # intersection of citations and k
    h_idx = np.max(np.minimum(citations, array))
    
    return h_idx

def get_h_index(data):
    '''Number of papers (h) that have received at least h citations'''
    h_indexes = []
    for a in np.unique(data['AU-ID']):
        citation_counts = data[(data['AU-ID']==int(a))]['citedby-count']
        try:
            h_indexes.append(h_index_expert(citation_counts))
        except ValueError:
            h_indexes.append(1)
    
    fig = plt.figure(facecolor='w')
    plt.hist(h_indexes, bins=int(len(data)/1000), histtype='step');
    plt.xlabel('h-index');
    print('Averange h-index: ', round(np.mean(h_indexes), 1))
    print('Standard deviation:', round(np.std(h_indexes), 2))
       
        
def get_last_year_published(data):
    '''Number of papers (h) that have received at least h citations'''
    last_years = []
    for a in np.unique(data['AU-ID']):
        last_year = np.max(data[(data['AU-ID']==int(a))]['Year'])
        last_years.append(last_year)
    
    fig = plt.figure(facecolor='w')
    plt.hist(last_years, bins=int(len(data)/1000), histtype='step');
    plt.xlabel('last year published');
    print('Averange last year published: ', round(np.mean(last_years), 1))
    print('Standard deviation:', round(np.std(last_years), 2))
    
def stats_per_year(data, y, ytype='Year'):
    n_papers=[]; n1_papers=[]; n_citations=[]; n_authors=[]
    none_count=0
    for a in np.unique(data['AU-ID']):
        papers = data[((data['AU-ID']==int(a)) & (data[ytype]==float(y)))]
        if len(papers)>0:
            n_papers.append(len(papers))
            n1_papers.append(len(papers[(papers['first author']==int(a))]))
            n_citations.append(np.sum(papers['citedby-count'])) # total citations received by this author for publication from this year
            n_authors.append(np.mean(papers['n authors']))
        else:
            none_count+=1
            #print('No papers found for:', a)
    #print(np.mean(papers['citedby-count'])) 
    return n_papers, n1_papers, n_citations, n_authors


def aggregate_data(data, save_path, years=np.arange(1995, 2022, 1), ytype='Year', status=False):
    n_by_year = []; n_by_year_e = []
    n1_by_year=[]; n1_by_year_e=[]
    weighted=[]; weighted_e=[]
    cit = []; cit_e = []
    auth = []; auth_e = []

    for y in years:
        if status==True:
            print('Working on', y)
        paps, paps1, cits, auths = stats_per_year(data, y=y, ytype=ytype)
        
        slp = np.sqrt(len(paps))
        n_by_year.append(np.mean(paps))
        n_by_year_e.append(np.std(paps) / slp)
        n1_by_year.append(np.mean(paps1))
        n1_by_year_e.append(np.std(paps1) / np.sqrt(len(paps1)))
        
        weighted_av = np.average(paps, weights=cits)
        weighted.append(weighted_av)
        weighted_variance = np.average((paps-weighted_av)**2, weights=cits)
        weighted_e.append(np.sqrt(weighted_variance) / slp)
        
        cit.append(np.mean(cits)); cit_e.append(np.std(cits) / slp)
        auth.append(np.mean(auths)); auth_e.append(np.std(auths) / slp)
        
    n_papers = pd.DataFrame(data={'year':years, 
                                  'av_n_papers':n_by_year, 'av_n_papers_e':n_by_year_e,
                                  'av_n_first_author_papers':n1_by_year, 'av_n_first_author_papers_e':n1_by_year_e,
                                  'n_papers_weighted':weighted, 'n_papers_weighted_e':weighted_e,
                                  'av_n_citations':cit, 'av_n_citations_e':cit_e, 
                                  'av_n_authors':auth, 'av_n_authors_e':auth_e})
    n_papers.to_csv(save_path)
    print('aggregate data saved to ',save_path)
    
def plot_aggregate_data(file_path, stat, normed=False):
    stats = pd.read_csv(file_path)
    p=1
    if normed==True:
        p = np.sum(stats[stat]) / len(stats['year'])
        plt.ylabel('Normalized Stat')
    plt.errorbar(stats['year'], stats[stat]/p, yerr=stats[stat+'_e']/p, label=stat, 
                 capsize=2, linewidth=1)
    plt.xlabel('Year'); 
