from collections import OrderedDict
from evaluate import RESULT_FOLDER
import os
import pandas as pd
import re


def combine(iter_rgb, iter_depth):
    result_folder = os.path.join(RESULT_FOLDER, 'confusion', 'RGB_' + str(iter_rgb) + ' ' + 'Depth_' + str(iter_depth))
    train_test_combinations = OrderedDict()
    roots = []
    for root, _, files in os.walk(result_folder):
        for file in files:
            if file == 'result.csv':
                roots.append(root)
    roots.sort()
    for root in roots:
        file = 'result.csv'
        delim_positions = [m.start() for m in re.finditer('_', os.path.basename(root))]
        train_test = os.path.basename(root)[:delim_positions[1]]
        if train_test in train_test_combinations.keys():
            train_test_combinations[train_test].append(
                {'file': os.path.join(root, file), 'kernels': os.path.basename(root)[delim_positions[1] + 1:]})
        else:
            train_test_combinations[train_test] = [
                {'file': os.path.join(root, file), 'kernels': os.path.basename(root)[delim_positions[1] + 1:]}]

    for train_test in train_test_combinations.keys():
        results = train_test_combinations[train_test]
        dfs = []
        for result in results:
            df = pd.read_csv(result['file'], index_col=0)
            df.loc['kernels'] = pd.Series()
            df = df.reindex(['kernels'] + [ind for ind in df.index if ind != 'kernels'])
            df = df.rename({'Overall': 'OVERALL', 'kernels': 'KERNELS: ' + result['kernels']})
            df.fillna('')
            dfs.append(df)
        combined_result = pd.concat(dfs, axis=0)
        combined_result.fillna('')
        print(train_test)
        combined_result.to_csv(os.path.join(result_folder, train_test + '_results.csv'))


if __name__ == '__main__':
    combine(800, 800)
