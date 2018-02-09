import numpy as np
import pandas as pd

from matplotlib import pyplot as plt 
import numpy as np  
data = pd.read_csv('dif.csv')
diffs = data['dif']
bins = [i*100-2000 for i in range(41)]
    
plt.hist(diffs, bins = bins) 
plt.xlabel('Number of place moved')
plt.ylabel('Frequency')
plt.title("Shake-up") 
plt.show()
#plt.save_fig('dif.png')