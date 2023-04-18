import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
import numpy as np

import sys
sys.path.insert(1, '/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/utils/')
from utils import load_data

df = pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/results_ucr.csv')


plt.figure(figsize=(6,6))
plt.plot([0,1],[0,1],c='black')
plt.title('ROCKET-500+HCF-6 vs ROCKET-500+HCF-6+Inception-ppv+max+GAP', fontsize=12)
plt.xlabel('ROCKET-500+HCF-6', fontsize=12)
plt.ylabel('ROCKET-500+HCF-6+Inception-ppv+max+GAP', fontsize=12)
plt.text(0.4,0.2,'ROCKET-500+\nHCF-6 is better', fontsize=12)
plt.text(0,0.65,'ROCKET-500+HCF-6+Inception\n-ppv+max+GAP is better',fontsize=12)

win = df.loc[df.iloc[:,1] < df.iloc[:,2]]
tie = df.loc[df.iloc[:,1] == df.iloc[:,2]]
loss = df.loc[df.iloc[:,1] > df.iloc[:,2]]

plt.scatter(win.iloc[:,1], win.iloc[:,2], c='blue', label=f'Win: {len(win)}')
plt.scatter(tie.iloc[:,1], tie.iloc[:,2], c='green', label=f'Tie: {len(tie)}')
plt.scatter(loss.iloc[:,1], loss.iloc[:,2], c='orange', label=f'Loss {len(loss)}')

plt.legend(fontsize=12)
plt.savefig('accuracies difference.png')
