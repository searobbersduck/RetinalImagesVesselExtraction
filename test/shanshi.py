import pandas as pd

ann_file = 'shangshi.csv'

df = pd.DataFrame.from_csv(ann_file)

row_list = []
for index, row in df.iterrows():
    row_list.append(row)

# print(row_list)

dr_ss_list = {}

for l in row_list:
    try:
        if l.name.find('.jpeg') >= 0:
            dr_ss_list[l.name] = l[0]
    except:
        continue

kaggle_file = 'trainLabels.csv'
kaggle_df = pd.DataFrame.from_csv(kaggle_file)
row_list = []
for index, row in kaggle_df.iterrows():
    row_list.append(row)

# print(row_list)

same_cnt = 0
unsame_cnt = 0

same_0 = 0
same_1 = 0
same_2 = 0
same_3 = 0
same_4 = 0
unsame_notsure = 0
unsame_laser = 0

name_list = []
ss_list = []
k_list = []

for l in row_list:
    name = l.name+'.jpeg'
    if name in dr_ss_list:
        dr_ss = dr_ss_list[name]
        if dr_ss == str(l[0]) and dr_ss == '0':
            same_0 +=1
        elif dr_ss == str(l[0]) and dr_ss == '1':
            same_1 +=1
        elif dr_ss == str(l[0]) and dr_ss == '2':
            same_2 +=1
        elif dr_ss == str(l[0]) and dr_ss == '3':
            same_3 +=1
        elif dr_ss == str(l[0]) and dr_ss == '4':
            same_4 +=1
        else:
            if dr_ss == '不确定':
                unsame_notsure += 1
            elif dr_ss == '激光治疗后':
                unsame_laser += 1
            else:
                unsame_cnt += 1
                name_list.append(name)
                ss_list.append(dr_ss)
                k_list.append(str(l[0]))

print('same: {}'.format(same_cnt))
print('same_0: {}'.format(same_0))
print('same_1: {}'.format(same_1))
print('same_2: {}'.format(same_2))
print('same_3: {}'.format(same_3))
print('same_4: {}'.format(same_4))
print('unsame_notsure: {}'.format(unsame_notsure))
print('unsame_laser: {}'.format(unsame_laser))
print('unsame: {}'.format(unsame_cnt))


import numpy as np
data = np.column_stack((name_list, ss_list, k_list))
data_df = pd.DataFrame(data, columns=['image', 'ss_dr', 'kaggle_dr'])
data_df.to_csv('compare.csv')
