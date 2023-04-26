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
plt.title('ROCKET-10000+HCF-6+Inception-ppv\nvs\nInception', fontsize=12)
plt.xlabel('ROCKET-10000+HCF-6+Inception-ppv', fontsize=12)
plt.ylabel('Inception', fontsize=12)
plt.text(0.4,0.23,'ROCKET-10000+HCF-6+Inception\n-ppv is better', fontsize=12)
plt.text(0.05,0.63,'Inception is better',fontsize=12)

plt.xlim([0, 1])
plt.ylim([0, 1])

win = diff.loc[df.iloc[:,4]  > diff.iloc[:,9]]
tie = diff.loc[df.iloc[:,4] == diff.iloc[:,9]]
los = diff.loc[df.iloc[:,4]  < diff.iloc[:,9]]
p_val = np.round(scipy.stats.wilcoxon(diff.iloc[:,9], diff.iloc[:,4])[1], 4)

plt.scatter(win.iloc[:,4], win.iloc[:,9], s=120, c='blue', label=f'Win: {len(win)}')
plt.scatter(tie.iloc[:,4], tie.iloc[:,9], s=120, c='green', label=f'Tie: {len(tie)}')
plt.scatter(los.iloc[:,4], los.iloc[:,9], s=120, c='orange', label=f'Loss {len(los)}')
plt.plot([], [], ' ', label="p_val: " + str(p_val))

plt.legend(fontsize=12)
plt.savefig('ppv vs inception.png')
