# coding:utf-8
import pandas as pd
from pandas import DataFrame


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode  # needs to be updated
        self.children = {}
        self.timeLink = []

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  ' * ind, self.name, '-', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def rollingEvents(df: DataFrame, timestamp_col_name: str, alarm_correlation_col_name: str,
                  window_size: int = 300, step_window_size: int = 120):
    events = []
    window_start_time = df[timestamp_col_name].min()
    window_step_start_time = window_start_time

    while window_step_start_time < window_start_time + window_size:
        this_start_window = window_step_start_time
        event_window = []

        for idx, row in df.iterrows():
            if row[timestamp_col_name] > this_start_window + window_size:
                events.append(event_window)
                event_window = [[row[alarm_correlation_col_name], 1, row[timestamp_col_name]]]
                this_start_window += window_size
            else:
                exists_element = next(filter(lambda x: x[0] == row[alarm_correlation_col_name], event_window), None)
                if exists_element is None:
                    exists_element = [row[alarm_correlation_col_name], 1, row[timestamp_col_name]]
                    event_window.append(exists_element)
                else:
                    exists_element[1] += 1
        window_step_start_time += step_window_size
    return events


def createInitSet(dataSet):
    retDict = {}
    new_dataSet = [[t[0] for t in trans] for trans in dataSet]

    for trans in new_dataSet:
        tran = tuple(t for t in trans)
        if tran in retDict:
            retDict[tran] += 1
        else:
            retDict[tran] = 1
    return retDict


def createTree(dataSet, minSup=1):  # create FP-tree from dataset but don't mine
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    del_k = []
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del_k.append(k)
    # print('del key (< minsup):', del_k)
    for k in del_k:
        del headerTable[k]
    freqItemSet = set(headerTable.keys())
    # print('freqItemSet: ', freqItemSet)
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # print('headerTable: ', headerTable)
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # 这里保持时序，不对时序进行排序
            orderedItems = [v[0] for v in localD.items()]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            prefix_path = prefixPath[1:]
            prefix_path.reverse()
            condPats[tuple(prefix_path)] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def getNodeFreq(treeNode):
    count = 0
    while treeNode != None:
        count += treeNode.count
        treeNode = treeNode.nodeLink

    return count


def mineFrequencyTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        basePatCount = getNodeFreq(headerTable[basePat][1])
        newFreqSet.insert(0, (basePat, basePatCount))
        freqItemList.append(newFreqSet)

        condPattBases = findPrefixPath(headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)

        if myHead != None:
            mineFrequencyTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def generateRules(retFreqList, minCon=0.2):
    L1 = {(tran[0][0],): tran[0][1] for tran in retFreqList if len(tran) == 1}
    L2 = list([tran for tran in retFreqList if len(tran) == 2])
    L2 = {(trans[0][0], trans[1][0]): trans[0][1] for trans in L2}

    k = 2
    Lk = list([tran for tran in retFreqList if len(tran) == k])
    L = [L1]
    L_rules = []
    while len(Lk) > 0:
        Lk_count = {}
        Lk_rules = []
        for i in range(k - 1):
            for l in Lk:
                l_left = tuple(tran[0] for tran in l[:i + 1])
                l_right = tuple(tran[0] for tran in l[i + 1:])
                l_count = l[i][1]

                if l_left in L[i] and l_right in L[k - i - 2]:
                    l_left_count = L[i][l_left]
                    #l_right_count = L[k - i - 2][l_right]
                    con = l_count / l_left_count
                    if con >= minCon:
                        Lk_count[tuple(tran[0] for tran in l)]=l_count
                        Lk_rules.append((l_left, l_right))
        L.append(Lk_count)
        if len(Lk_rules) > 0:
            L_rules.append(Lk_rules)
        else:
            break
        k += 1
        Lk = list([tran for tran in retFreqList if len(tran) == k])

    # L2_con = {tran: float(L2[tran]) / float(L1[tran[0]]) for tran in L2.keys()}
    # return [tran for tran in L2_con.keys() if L2_con[tran] >= minCon]
    return L_rules


if __name__ == '__main__':
    df = pd.read_csv("../../data/Alarm.csv")
    event_time_data = rollingEvents(df, "start_timestamp", "alarm_id", 300, 120)
    dataSet = createInitSet(event_time_data)
    minSup = int(sum(dataSet.values()) * 0.2)
    minCon = 0.6

    # dataSet = {(1, 2, 3, 4): 50, (2, 3, 4): 1, (2, 3, 5): 1, (1, 2, 3): 1, (2, 3, 4, 5): 1, (1, 2, 3, 4, 5): 1,
    #            (2, 3, 1): 2, (2, 3, 4): 1, (2, 5): 10}
    # minSup = int(sum(dataSet.values()) * 0.1)
    # minSup = minSup if minSup > 2 else 2

    fpTree, headerTable = createTree(dataSet, minSup)
    # print(headerTable)
    fpTree.disp()
    # print(findPrefixPath(headerTable[5][1]))
    retFreqList = []
    mineFrequencyTree(fpTree, headerTable, minSup, [], retFreqList)
    print("freq items", retFreqList)
    rules = generateRules(retFreqList, minCon)
    print(len(rules))
    for i in range(len(rules)):
        print("=" * 30)
        print("level:", i+1)
        for rule in rules[i]:
            print(rule[0], "->", rule[1])
