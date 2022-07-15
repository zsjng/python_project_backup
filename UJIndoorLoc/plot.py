import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv('res-hn.txt', sep='\t')
# df_T = pd.DataFrame(data.values.T,columns=data.rows,index=data.colums)
# data.T.plot(kind='bar')
# labels = data.columns
labels = []
mean = []
std = []
for head in data.columns:
    labels.append(head[:3])
    mean.append(np.mean(data[head]))
    std.append(np.std(data[head]))

x = np.arange(len(data.columns))  # the label locations
width = 0.5  # the width of the bars
#
fig, ax = plt.subplots()
plt.ylim(0.9, 0.96)
rects1 = ax.bar(x, mean, width, color='#87CEFA', ec='black')
plt.errorbar(x, mean, yerr=std, fmt='none', ecolor='gray', elinewidth=1.2, capthick=1.2, capsize=4)
# rects2 = ax.bar(x + width/2, PLCC, width, label='PLCC', color='#B0E0E6', ec='black')
# plt.errorbar(x + width/2, PLCC, yerr=PLCC_std, fmt='none', ecolor='gray', elinewidth=1.2, capthick=1.2, capsize=4)
#
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Hidden nodes')
plt.ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
# plt.grid(linestyle="--", alpha=0.3, zorder=-1)
# # ax.set_title('Scores by group and gender')
# ax.legend()
plt.show()
