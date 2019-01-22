import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == '__main__':
	DATAFILE = 'faithful.dat'
	
	df = pd.read_table(DATAFILE, header=12)
	
