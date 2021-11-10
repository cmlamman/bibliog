import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, math
from scipy import stats
import pycountry_convert as pc

def n_authors(data):
	nauthors = [len(a.split(';')) for a in data['authors']]
	plt.hist(nauthors, bins=51);
	plt.xlabel('n authors');
	print('Mean: ', round(np.mean(nauthors)), '+/- ', round(np.std(nauthors)))
