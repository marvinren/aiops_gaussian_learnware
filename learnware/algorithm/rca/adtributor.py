import pandas as pd
import numpy as np
import scipy.stats


def js_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    m = (p + q) / 2

    # 方法一：自定义函数
    js1 = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    # 方法二：调用scipy包
    js2 = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)

    return round(float(js1), 6)


class MultiDimRCA:

    def __init__(self, teen=0.1, tep=0.7):
        self.teen = teen
        self.tep = tep

    def predict(self, data):
        value_sum = data.sum(numeric_only=True)
        ExplanatorySet = []
        for i in data.columns:
            # 对于f和v值不进行计算
            if i == 'f' or i == 'v':
                continue
            # dim = data[i].unique()
            # 先计算isp维度的
            group_dim = data.groupby(i).sum()
            group_dim = group_dim.reset_index()
            group_dim['f_sum'] = value_sum['f']
            group_dim['v_sum'] = value_sum['v']

            group_dim['p'] = group_dim['f'] / group_dim['f_sum']
            group_dim['q'] = group_dim['v'] / group_dim['v_sum']

            group_dim['surprise'] = group_dim[['p', 'q']].apply(lambda x: js_divergence(x['p'], x['q']), axis=1)
            group_dim['EP'] = group_dim[['f', 'v', 'f_sum', 'v_sum']].apply(
                lambda x: (x['v'] - x['f']) / (x['v_sum'] - x['f_sum']), axis=1)

            ep_sum = group_dim['EP'].sum()
            t_ep = ep_sum * self.tep
            t_een = ep_sum * self.teen
            i_candidate = []
            i_explanatory = 0
            i_suprise = 0
            ep_dataframe = group_dim[group_dim['EP'] > t_een]

            for idx, j in ep_dataframe.sort_values(by=['surprise'], ascending=False).iterrows():
                i_explanatory += j['EP']
                i_suprise += j['surprise']
                i_candidate.append(j[i])
                if i_explanatory > t_ep:
                    ExplanatorySet.append([i, ",".join(i_candidate), i_explanatory, i_suprise])
                    break

        df = pd.DataFrame(ExplanatorySet, columns=['Dimension', 'Element', 'ExplanatoryPower', 'Surprise'])
        return df.sort_values(by=['Surprise'], ascending=False)


if __name__ == "__main__":

    # 量值分析
    # data = pd.DataFrame({
    #     "ISP": ["Mobile", "Mobile", "Mobile", "Unicom", "Unicom", "Unicom", "Unicom", "Mobile"],
    #     "Province": ["Beijing", "Shanghai", "Guangdong", "Beijing", "Shanghai", "Guangdong", "Tianjin", "Tianjin"],
    #     "Category": ["A", "B", "A", "B", "A", "B", "A", "B"],
    #     "f": [20, 15, 10, 10, 35, 40, 20, 15],
    #     "v": [22, 40, 42, 10, 35, 40, 23, 15]
    # })

    # 率值分析
    data = pd.DataFrame({
        "ISP": ["Mobile", "Mobile", "Mobile", "Unicom", "Unicom", "Unicom", "Unicom", "Mobile"],
        "Province": ["Beijing", "Shanghai", "Guangdong", "Beijing", "Shanghai", "Guangdong", "Tianjin", "Tianjin"],
        "Category": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "f": [100, 99, 99, 83, 83, 83, 83, 98],
        "v": [99, 99, 72, 82, 84, 82, 80, 93]
    })

    rca = MultiDimRCA()
    predict = rca.predict(data)
    print(predict)