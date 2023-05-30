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
plt.title('MultiROCKET-40000+HCF-6+Inception-ppv+lspv+mpv+mipv\nvs\nROCKET-10000+HCF-6+Inception-ppv+lspv+mpv+mipv+GAP+max', fontsize=12)
plt.xlabel('MultiROCKET-40000+HCF-6+Inception-\nppv+lspv+mpv+mipv', fontsize=12)
plt.ylabel('ROCKET-10000+HCF-6+Inception-\nppv+lspv+mpv+mipv+GAP+max', fontsize=12)
plt.text(0.32,0.18,'MultiROCKET-40000+HCF-6+Inception-\nppv+lspv+mpv+mipv is better', fontsize=11)
plt.text(0.02,0.58,'ROCKET-10000+HCF-6+Inception-\nppv+lspv+mpv+mipv\n+GAP+max is better',fontsize=11)

plt.xlim([0, 1])
plt.ylim([0, 1])

win = diff.loc[diff.iloc[:,2]  < diff.iloc[:,3]]
tie = diff.loc[diff.iloc[:,2] == diff.iloc[:,3]]
los = diff.loc[diff.iloc[:,2]  > diff.iloc[:,3]]
p_val = np.round(scipy.stats.wilcoxon(diff.iloc[:,2], diff.iloc[:,3])[1], 4)

plt.scatter(win.iloc[:,3], win.iloc[:,2], s=120, c='blue', label=f'Win: {len(win)}')
plt.scatter(tie.iloc[:,3], tie.iloc[:,2], s=120, c='green', label=f'Tie: {len(tie)}')
plt.scatter(los.iloc[:,3], los.iloc[:,2], s=120, c='orange', label=f'Loss {len(los)}')
plt.plot([], [], ' ', label="p_val: " + str(p_val))

plt.legend(fontsize=12)
plt.savefig('rock-ppv+lspv+mpv+mipv+GAP+max vs multi-ppv+lspv+mpv+mipv.png')
