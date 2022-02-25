import copy
import math

import pandas as pd
import numpy as np


class Node:
    def __init__(self):
        self.parent = None
        self.children = []
        self.combination = None

    def __repr__(self):
        return "(" + ",".join(self.combination) + ")"


def get_isolation_power(combination):
    o_avg_df = data.sum()
    x_avg_df = data.query(combination).mean()
    x_a = x_avg_df['f']
    x_b = x_avg_df['v']
    o_a = o_avg_df['f']
    o_b = o_avg_df['v']
    P_a_x = x_a / (x_a + x_b)
    P_b_x = x_b / (x_a + x_b)
    P_a_x_avg = (o_a - x_a) / (o_a + o_b - x_a - x_b)
    P_b_x_avg = (o_b - x_b) / (o_a + o_b - x_a - x_b)
    ip = 1 - (1 / (o_a + o_b)) * (
            x_a * math.log(1 / P_a_x) + x_b * math.log(1 / P_b_x) + (o_a - x_a) * math.log(1 / P_a_x_avg) + (
            o_b - x_b) * math.log(1 / P_b_x_avg))
    return ip


class iDice:

    def __init__(self, columns=[]):
        self.dimension = columns

    def get_dimensions(self):
        L = []
        for l in range(len(self.dimension)):
            this_L = []
            if l == 0:
                for i in range(len(self.dimension)):
                    this_L.append(tuple([i]))
            else:
                this_L = []
                for lp in L[l - 1]:
                    for l1 in L[0]:
                        if l1[0] in list(lp):
                            continue
                        new_c = copy.copy(list(lp))
                        new_c.append(l1[0])
                        new_c = sorted(new_c)
                        if tuple(new_c) in this_L:
                            continue
                        this_L.append(tuple(new_c))
            L.append(this_L)
        return L

    def fit_predict(self, data):
        root = Node()
        elements = [np.unique(data[d].values.tolist()) for d in self.dimension]
        self.generate_combinations_tree(root, elements)

        C = []
        for idx, row in data.iterrows():
            combinations = tuple([v for v in row[self.dimension]])
            # 1. Impact的剪枝
            if self.impact_punging(row['v']):
                continue
            # 2. Change Detection的剪枝
            if self.change_degree_pungine(row['v'], row['f']):
                continue
            C.append(combinations)


        # # 3. Isolation Power的剪枝
        # ip = self.isolation_power(combinations, L)
        # #this_L.append((combinations, ip))

    def impact_punging(self, d):
        return d < 2

    def change_degree_pungine(self, v, f):
        return abs(f - v) < 2 or float(abs(f - v)) / v < 0.1

    def isolation_power(self, combinations, L):
        asterisk_count = len(list(filter(lambda e: e == "*", list(combinations))))
        total_count = len(combinations)
        dimension_count = total_count - asterisk_count

        if dimension_count not in L.keys():
            L[dimension_count] = []

        if asterisk_count == 0:
            L[dimension_count].append(combinations)

    def generate_combinations_tree(self, node, elements):
        dimensions = self.dimension
        if node.combination is None:
            for i in range(len(dimensions)):
                for e in elements[i]:
                    child_node = Node()
                    child_node.parent = node
                    child_node.combination = ['*'] * len(dimensions)
                    child_node.combination[i] = e
                    node.children.append(child_node)
                    self.generate_combinations_tree(child_node, elements)
        else:
            asterisk_idx = [idx for idx, e in enumerate(node.combination) if e == "*"]
            if len(asterisk_idx) == 0:
                return
            for idx in asterisk_idx:
                for e in elements[idx]:
                    child_node = Node()
                    child_node.combination = copy.copy(node.combination)
                    child_node.combination[idx] = e
                    child_node.parent = node
                    node.children.append(child_node)
                    self.generate_combinations_tree(child_node, elements)



if __name__ == '__main__':
    data = pd.DataFrame({
        "ISP": ["Mobile", "Mobile", "Mobile", "Unicom", "Unicom", "Unicom"],
        "Province": ["Beijing", "Shanghai", "Guangdong", "Beijing", "Shanghai", "Guangdong"],
        "f": [20, 15, 10, 10, 25, 20],
        "v": [28, 15, 10, 10, 25, 20]
        # "v": [14, 9, 10, 7, 15, 20]
    })

    model = iDice(columns=['ISP', 'Province'])
    model.fit_predict(data)

    # print(get_isolation_power("ISP=='Mobile'"))
    # print(get_isolation_power("ISP=='Unicom'"))
    # print(get_isolation_power("ISP=='Mobile' and Province=='Beijing'"))
    # print(get_isolation_power("ISP=='Mobile' and Province=='Guangdong'"))
