import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

df = pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/results_ucr.csv')

inc = pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/inc.csv').iloc[:,[2,5]]
inc = inc.groupby('dataset_name').mean()

rock = pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/rock.csv').iloc[:,[0,1]]

diff = df.merge(rock, how='left', on='dataset')

diff = diff.merge(inc, how='left', left_on='dataset', right_on='dataset_name')

plt.figure(figsize=(6,6))
plt.plot([0,1],[0,1],c='black')
plt.title('ROCKET-500+HCF-6+Inception-ppv+max\nvs\nROCKET-500+HCF-6+Inception-GAP', fontsize=12)
plt.xlabel('ROCKET-500+HCF-6+Inception-ppv+max', fontsize=12)
plt.ylabel('ROCKET-500+HCF-6+Inception-GAP', fontsize=12)
plt.text(0.4,0.23,'ROCKET-500+HCF-6+Inception\n-ppv+max is better', fontsize=12)
plt.text(0.05,0.63,'ROCKET-500+HCF-6+Inception\n-GAP is better',fontsize=12)

plt.xlim([0, 1])
plt.ylim([0, 1])

win = diff.loc[df.iloc[:,1]  > diff.iloc[:,6]]
tie = diff.loc[df.iloc[:,1] == diff.iloc[:,6]]
los = diff.loc[df.iloc[:,1]  < diff.iloc[:,6]]
p_val = np.round(scipy.stats.wilcoxon(diff.iloc[:,6], diff.iloc[:,1])[1], 4)

plt.scatter(win.iloc[:,1], win.iloc[:,6], s=120, c='blue', label=f'Win: {len(win)}')
plt.scatter(tie.iloc[:,1], tie.iloc[:,6], s=120, c='green', label=f'Tie: {len(tie)}')
plt.scatter(los.iloc[:,1], los.iloc[:,6], s=120, c='orange', label=f'Loss {len(los)}')
plt.plot([], [], ' ', label="p_val: " + str(p_val))

plt.legend(fontsize=12)
plt.savefig('ppv+max vs GAP.png')
