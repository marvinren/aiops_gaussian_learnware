import pandas as pd
import os


if __name__ == "__main__":

    error_ts = "1544850900000"

    base_dir = "/Users/renzhiqiang/Workspace/data/MultiDimension-Localization/part1"
    pathDir = os.listdir(base_dir)

    data = None
    num = 0
    for allDir in pathDir:
        df = pd.read_csv(os.path.join(base_dir, allDir),
                         names=['A', 'B', 'C', 'D', 'E', 'v'],
                         index_col=['A', 'B', 'C', 'D', 'E'])
        if data is None:
            data = df
            data['f'] = df['v']
        else:
            data['f'] += df['v']
        num += 1
        if allDir.startswith(error_ts):
            data['f'] /= num
            break

    data.to_csv("/Users/renzhiqiang/Workspace/data/root.csv")