import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

df = pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/results_ucr.csv')

multi =  pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/multi_rock.csv').iloc[:,[0,-1]]

inc = pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/inc.csv').iloc[:,[2,5]]
inc = inc.groupby('dataset_name').mean()

rock = pd.read_csv('/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/results/rock.csv').iloc[:,[0,1]]

diff = df.merge(rock, how='left', on='dataset')

diff = diff.merge(inc, how='left', left_on='dataset', right_on='dataset_name')

diff = diff.merge(multi, how='left', left_on='dataset', right_on='dataset_name')

diff.drop(['dataset_name'], axis=1, inplace=True)

diff.to_csv('all_results.csv', index=False)

plt.figure(figsize=(6,6))
plt.plot([0,1],[0,1],c='black')
plt.title('RMultiROCKET-40000+HCF-6+Inception-ppv\nvs\nInception', fontsize=12)
plt.xlabel('MultiROCKET-40000+HCF-6+Inception\n-ppv', fontsize=12)
plt.ylabel('Inception', fontsize=12)
plt.text(0.4,0.23,'MultiROCKET-40000+HCF-6+Inception\n-ppv', fontsize=12)
plt.text(0.05,0.63,'Inception is better',fontsize=12)

plt.xlim([0, 1])
plt.ylim([0, 1])

win = diff.loc[diff.iloc[:,-2]  < diff.iloc[:,3]]
tie = diff.loc[diff.iloc[:,-2] == diff.iloc[:,3]]
los = diff.loc[diff.iloc[:,-2]  > diff.iloc[:,3]]
p_val = np.round(scipy.stats.wilcoxon(diff.iloc[:,-2], diff.iloc[:,3])[1], 4)

plt.scatter(win.iloc[:,3], win.iloc[:,-2], s=120, c='blue', label=f'Win: {len(win)}')
plt.scatter(tie.iloc[:,3], tie.iloc[:,-2], s=120, c='green', label=f'Tie: {len(tie)}')
plt.scatter(los.iloc[:,3], los.iloc[:,-2], s=120, c='orange', label=f'Loss {len(los)}')
plt.plot([], [], ' ', label="p_val: " + str(p_val))

plt.legend(fontsize=12)
#plt.savefig('ppv vs Inception.png')
