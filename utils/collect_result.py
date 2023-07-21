
from collections import Counter
import pandas as pd
import os
import re
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def get_index(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]
    
def path_to_csv(filepath='./exp', criterion=['accuracy', 'precision', 'recall', 'f1'], evaluate=['f1'], 
    largest=5, retrun=5, logname='test.log', csvfile='test.csv', overwrite=False, wantlow=False, epoch=None):
    '''
    Record the average cross-validation result to a csv file
    if wantlow is True, the smaller the value is, the better.
    
    Input
    - filepath: path to every folds
    '''
    
    all_file_result = []
    files = os.listdir(filepath)
    for file in files:
        result = {c:[] for c in criterion}
        if epoch is not None:
            with open(os.path.join(filepath, file, 'Log', logname), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if f'Testing epoch: {epoch},' in line:
                        for c in criterion:
                            score = re.search(f' {c}: \d+.\d+', line).group()
                            score = float(re.sub(f' {c}: ', '', score))
                            result[c].append(score)
                            assert len(result[c]) == 1
                f.close()
            best_result = [result[c][0] for c in criterion]
        else:
            max_id = []
            for c in criterion:
                with open(os.path.join(filepath, file, 'Log', logname), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if c in line and 'workshop' not in line:
                            score = re.search(f' {c}: \d+.\d+', line).group()
                            score = float(re.sub(f' {c}: ', '', score))
                            result[c].append(score)
                    f.close()
                if c in evaluate:
                    result_set = set(result[c])
                    result_max = sorted(result_set, reverse=not wantlow)[:largest]
                    temp = []
                    for maximum in result_max:
                        temp += get_index(result[c], maximum) 
                    max_id.extend(temp)

            c = Counter(max_id)
            return_id_counts = c.most_common(retrun)

            if not wantlow:
                best_idx = 0
                best_sum = 0
                for idx, counts in return_id_counts:
                    s = sum([result[c][idx] for c in evaluate])  # criterion -> evaluate
                    if s > best_sum:
                        best_sum = s
                        best_idx = idx
            else:
                best_idx = 0
                best_sum = 10000
                for idx, counts in return_id_counts:
                    s = sum([result[c][idx] for c in evaluate])  # criterion -> evaluate
                    if s < best_sum:
                        best_sum = s
                        best_idx = idx
            best_result = [result[c][best_idx] for c in criterion]
        all_file_result.append(best_result)

    print('Calculate mean result from {} files. Write to {}'.format(len(all_file_result), csvfile))
    print(f'Evaluate: {evaluate}')
    mean_result = np.mean(np.array(all_file_result), axis=0).tolist()

    if not os.path.exists(os.path.dirname(csvfile)):
        os.makedirs(os.path.dirname(csvfile))
        
    if not os.path.exists(csvfile):
        data = {'Model': []}
        data.update({c: [] for c in criterion})
        df = pd.DataFrame(data)
        df.to_csv(csvfile, index=False, sep=',')

    newdata = {'Model': filepath[6:]}   # pass ./exp/
    newdata.update({c: [r] for c, r in zip(criterion, mean_result)})
    new_df = pd.DataFrame(newdata)
    if overwrite:
        df = new_df
    else:
        df = pd.read_csv(csvfile)
        df_temp = df[df['Model'] == new_df.loc[0, 'Model']]
        if df_temp.empty:
            df = pd.concat([df, new_df], ignore_index=True)   # insert a new line in DataFrame 
        else:
            row_index = df_temp.index.tolist()[0]
            for c in criterion:
                df.loc[row_index, c] = new_df.loc[0, c]

    for c in criterion:
        df[c] = df[c].apply(lambda x: round(x, 3))
    df.to_csv(csvfile, index=False, sep=',')
    tidy_csvfile(csvfile, colname='Model')

def plot_process(x: list, title: list, savedir: str):
    col = math.ceil(len(x) / 2)
    assert col < 5, print('Get too many data, the maximun number of columns in figure is 4.')
    line = 2
    
    color = ['b', 'g', 'k', 'r']
    color = color[:col] * line
    plt.figure(figsize=(18, 8))
    plt.suptitle(savedir.split('/', maxsplit=1)[-1])
    
    plt.subplots_adjust(wspace=0.15, hspace=0.3, bottom=0.2) 
    for i, (data_x, data_title, c) in enumerate(zip(x, title, color)):
        y = np.arange(len(data_x))
        plt.subplot(line, col, i + 1)
        plt.plot(y, data_x, c)
        plt.title(data_title)

    plt.savefig(os.path.join(savedir, 'result.png'), bbox_inches='tight', pad_inches=0.2)
    plt.close()

    return

def tidy_csvfile(csvfile, colname, ascending=True):
    '''
    tidy csv file base on a particular column.
    '''
    print(f'tidy file: {csvfile}, base on column: {colname}')
    df = pd.read_csv(csvfile)
    df = df.sort_values(by=[colname], ascending=ascending, na_position='last')
    df = df.round(3)
    df.to_csv(csvfile, index=False, sep=',')

