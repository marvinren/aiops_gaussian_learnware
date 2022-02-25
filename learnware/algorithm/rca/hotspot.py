import copy
import math
import operator
import random

import pandas as pd
import numpy as np
from pandas import DataFrame


class Node(object):
    """
    MCTS树，这里将连接线的N，Q值也写入节点中
    """

    # node类初始化
    def __init__(self):
        self.parents = None
        self.children = []
        self.state = []

        self.Q = 0
        self.N = 0

    def __repr__(self):
        return f"{self.state}: Q({self.Q}) N({self.N}) C({len(self.children)})"


class HotSpot:
    """
    HotSpot算法，详情请参照：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8288614
    dimension：所有维度
    max_search_num: M值，最大搜索次数
    q_threshold:PT值，q值阈值，作为搜索的停止条件
    """

    def __init__(self, dimensions: list, max_search_num=1000, q_threshold=0.75):
        # 最大的搜索次数
        self.max_search_num = max_search_num
        # Q值的阈值
        self.q_threshold = q_threshold
        # 用于punge的Q值阈值
        self.min_q_threshold = 1e-5
        # 所有维度
        self.dimension = dimensions

    def fit_predict(self, data: DataFrame):

        Layer_num = len(self.dimension)

        L = []
        L_elements = []
        result_list = []
        for l in range(Layer_num):
            this_L = self.get_dim_combinations(L, l)
            this_elements = {}
            print("=" * 30)
            print("开始维度层{}, 含有维度{}".format((l + 1), (this_L)))

            for cuboid in this_L:
                dims = list(cuboid)
                elements = []
                if len(dims) == 1:
                    elements = np.unique(data[dims].values).tolist()
                else:
                    print(L_elements[l - 2])
                    # print(len(L_elements[l-1]))
                    if len(L_elements[l - 2]) < 2:
                        break
                    left = L_elements[l - 1][",".join(sorted(dims[:-1]))]
                    right = L_elements[0][dims[-1]]
                    elements = [[le, re] if type(le) is not list else le + [re] for le in left for re in right]
                print("针对维度{}中，发现有元素{}, 开始搜索.....".format(dims, elements))
                best_node = self.MCTS(data, dims, elements)
                print(best_node)
                if best_node is not None and len(best_node.state) > 0:
                    this_elements[",".join(sorted(dims))] = best_node.state
                    result_list.append((best_node.state, best_node.Q))
            # 将punge后的维度组合，接入本层维度 (?)
            L.append(this_L)
            L_elements.append(this_elements)
            print("结果如下：")
            print(this_L)
            print(this_elements)
        return list(reversed(sorted(result_list, key=lambda e: e[1])))

    def get_dim_combinations(self, L, l):
        # 生成本层的维度组合
        if l == 0:
            this_L = [set([d]) for d in self.dimension]
        else:
            this_L = []
            for pl in L[l - 1]:
                for l1 in L[0]:
                    if not l1.issubset(pl):
                        set_e = pl.copy().union(l1)
                        if set_e not in this_L:
                            this_L.append(set_e)
        return this_L

    def MCTS(self, data, dims, elements):
        # 计算单节点
        score_single_e = []
        for e in elements:
            e, ps_score = self.get_q_scores(data, dims, e)
            print("MCTS: 计算{}的ps值{}".format(e, ps_score))
            score_single_e.append(ps_score)

        # print(score_single_e)
        # 初始化，创建根节点
        node = Node()
        max_q = 0
        best_node = None

        # 开始搜索，最大搜索次数可变
        for n in range(self.max_search_num):
            # 1. 选择，如果所有节点搜索完毕，则跳出循环
            selection_node, all_selected = self.selection(node, elements, score_single_e)
            print("MCTS:1.selection, 选择节点{}, 是否选择穷尽{}".format(selection_node, all_selected))
            # 如果所有元素都已经被选择过，整颗树搜索完毕了，达到终止条件，跳出
            if all_selected:
                break

            # 2、扩展，获得剩余元素中的最大元素值
            max_e, max_s = self.expansion(selection_node, score_single_e, elements)
            print("MCTS:2.expand：基于节点{}准备扩展节点为 {}".format(selection_node.state, max_e))

            # 3、评价，原状态与最大元素值组合成新状态，获得新状态的Q值
            if max_e == -1:
                new_q = 0
            else:
                new_q = self.evalation(selection_node, max_e, data, dims)
            print("MCTS:3.evalation: 获取新的Q值为 {}".format(new_q))

            # 4、更新，新状态节点值跟节点路径中的每个节点：N+1, Q为路径中最大的Q值
            self.backup(selection_node, max_e, new_q)
            print("MCTS:4.backup")

            # 如果根节点Q值变大，则更新最优节点
            if node.Q > max_q:
                best_node = self.get_best_node(node)
                max_q = node.Q
            # 如果新节点的Q值超过预设阀值，则跳出循环
            if new_q >= self.q_threshold:
                break

            print("best_node", best_node)
        return best_node

    def get_q_scores(self, data, dims, e):
        a = []
        query_str = []
        if type(e) is str:
            e = [e]
        if type(e[0]) is list:
            for element in e:
                query_l = " and ".join(
                    [dims[di] + "=='" + (element[di] if type(element) is list else element) + "'" for di in
                     range(len(dims))])
                query_str.append(query_l)
        else:
            query_str = [dims[di] + "=='" + e[di] + "'" for di in range(len(dims))]
        single_data = data.query(" or ".join(query_str))
        sum_f = single_data['f'].sum()
        sum_v = single_data['v'].sum()
        for idx, row in data.iterrows():
            row_dim = [row[dim] for dim in dims]
            if type(e[0]) is str and (operator.eq(row_dim, e) or set(row_dim).issubset(e)):
                if len(dims) == len(self.dimension):
                    a.append(row['v'])
                else:
                    a.append(self.getValueA(row['f'], sum_f, sum_v))
            elif type(e[0]) is list:
                flag = False
                for element in e:
                    if operator.eq(row_dim, element) or set(row_dim).issubset(element):
                        flag = True
                        break
                if flag:
                    if len(dims) == len(self.dimension):
                        a.append(row['v'])
                    else:
                        a.append(self.getValueA(row['f'], sum_f, sum_v))
                else:
                    a.append(row['f'])
            else:
                a.append(row['f'])
        v = data["v"].values
        f = data["f"].values
        ps_score = max(1 - self.getDistance(v, a) / self.getDistance(v, f), 0)
        return e, ps_score

    def getValueA(self, fi, f, v):
        # Ripple Effect 计算a值
        return fi - float(fi) * (f - v) / f

    def getDistance(self, u, w):
        # 计算两向量的距离(欧氏距离)
        return math.sqrt(np.sum((u - w) ** 2))

    def selection(self, node, elements, score_single_e):
        all_selected = False
        # 每次选择都是从根节点选择Q最大的进行扩展，有一定几率选择维未被选择到的节点
        # 这里本来要全部扫描完的数据作为结束条件, 由于存在剪枝，那么只要吧所有大于阈值的元素扫描完，即为结束
        while len(node.state) < ((np.array(score_single_e) >= self.min_q_threshold).sum()):
            if len(node.children) == 0:
                self.init_children(node, elements, score_single_e)
            # 如果当前节点存在没有访问过的孩子节点，则依据概率选择深度优先还是广度优先
            Q_max = 0
            is_random = False
            for i in node.children:
                if i.Q > Q_max:
                    Q_max = i.Q
                if i.N == 0:
                    is_random = True

            # BFS
            if is_random:
                if random.random() > Q_max:
                    return node, all_selected

            # 否则依据UCB公式计算最优的孩子节点，重复这个过程
            node = self.best_child(node)
            if node is None:
                break
        # 如果节点包括了所有元素，那么达到了终止条件
        all_selected = True
        return node, all_selected

    def init_children(self, node, elements, score_single_e):
        rest_e = [i for i in elements if i not in node.state]
        for e in rest_e:
            idx = elements.index(e)
            # 对小于最小阈值的元素进行剪枝
            if score_single_e[idx] < self.min_q_threshold:
                continue
            child = Node()
            child.state = copy.deepcopy(node.state)
            child.state.append(e)
            child.parents = node
            node.children.append(child)

    def best_child(self, node):
        # 依据UCB公式计算最优孩子节点
        best_score = -1
        best = None

        for sub_node in node.children:

            # 在可选的节点里面选择最优
            if sub_node.Q > 0:
                C = math.sqrt(2.0)
                left = sub_node.Q
                right = math.log(node.N) / sub_node.N
                score = left + C * math.sqrt(right)

                if score > best_score:
                    best = sub_node
                    best_score = score

        return best

    def expansion(self, selection_node, score_single_e, elements):
        # 得到所有孩子节点中的新元素
        e_field = []
        for i in selection_node.children:
            if i.N == 0:
                e_field.append(i.state[-1])

        # 在新元素中选择Q值最大的一个
        max_e, max_score = self.get_max_e(e_field, score_single_e, elements)
        return max_e, max_score

    def get_max_e(self, e_field, score_single_e, elements):
        # 获得可选元素中PS值最大的
        max_e = -1
        max_score = -1
        for e in e_field:
            # 避免重复计算，score_single_e在主函数中计算
            if type(elements) is np.ndarray:
                elements = elements.tolist()
            index = elements.index(e)
            score = score_single_e[index]
            if score > max_score:
                max_score = score
                max_e = e
        return max_e, max_score

    def evalation(self, selection_node, max_e, data, dims):
        new_set = copy.deepcopy(selection_node.state)
        new_set.append(max_e)

        e, ps_scores = self.get_q_scores(data, dims, new_set)
        return ps_scores

    def backup(self, selection_node, max_e, new_q):
        node = None
        for n in selection_node.children:
            if n.state[-1] == max_e:
                node = n
                break
        while node is not None:
            node.N += 1
            if new_q > node.Q:
                node.Q = new_q
            node = node.parents

    def get_best_node(self, node):
        best_score = node.Q
        while len(node.children) is not 0:
            for child in node.children:
                if child.Q == best_score:
                    node = child
                    break
        return node


if __name__ == "__main__":
    # data = pd.DataFrame({
    #     "ISP": ["Mobile", "Mobile", "Mobile", "Mobile", "Mobile", "Mobile", "Mobile", "Mobile", "Mobile", "Mobile",
    #             "Unicom", "Unicom", "Unicom", "Unicom", "Unicom", "Unicom", "Unicom", "Unicom", "Unicom", "Unicom"],
    #     "Province": ["Beijing", "Shanghai", "Tianjin", "Guangzhou", "Beijing", "Shanghai", "Tianjin", "Guangzhou",
    #                  "Beijing", "Shanghai",
    #                  "Beijing", "Shanghai", "Tianjin", "Guangzhou", "Beijing", "Shanghai", "Tianjin", "Guangzhou",
    #                  "Beijing", "Shanghai"],
    #     "Channel": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
    #     "v": [20, 15, 10, 10, 25, 20, 20, 15, 10, 10, 25, 20, 20, 15, 10, 10, 25, 20, 20, 15],
    #     "f": [24, 13, 15, 16, 32, 33, 21, 16, 12, 11, 25, 21, 21, 16, 11, 11, 26, 21, 21, 16]
    # })
    # model = HotSpot(["ISP", "Province", "Channel"], 10, 0.75)
    # print(model.fit_predict(data))

    data = pd.DataFrame({
        "ISP": ["Mobile", "Mobile", "Mobile", "Unicom", "Unicom", "Unicom"],
        "Province": ["Beijing", "Shanghai", "Guangdong", "Beijing", "Shanghai", "Guangdong"],
        "f": [20, 15, 10, 10, 25, 20],
        "v": [14, 9, 10, 7, 15, 20]
    })
    model = HotSpot(["Province", "ISP"], 100, 0.75)
    result = model.fit_predict(data)
    print(result)

    # data = pd.read_csv("/Users/renzhiqiang/Workspace/data/root.csv")
    # model = HotSpot(["A", "B", "C", "D"], 100)
    # print(model.fit_predict(data))
