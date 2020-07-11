import DataPre as dp
import numpy as np
import copy
import math


def userProcess(userCost, userTaskSet, totalUserNum):
    """
    计算每个user的task集合的大小
    :param userTaskSet: user task set
    :return: 分组；及每个user task的size
    """

    # 计算每个user的集合task size
    size = np.zeros((totalUserNum,), dtype=np.int)

    for i in range(totalUserNum):
        size[i] = sum(userTaskSet[:, i])
    userTaskSize = size

    # 计算每个用户的报价
    userBid = userTaskSize * userCost

    # # 对每个user的size进行分组
    # disSequence = []
    # for i in range(userTaskNumDis+1):
    #     disSequence.append(userTaskNumDis - i )
    # print("任务集合大小序列为：",disSequence)
    # groupDict = dict.fromkeys(disSequence, [])
    #
    # for i in range(totalUserNum):
    #     list = copy.deepcopy(groupDict[size[i]])
    #     list.append(i)
    #     groupDict[size[i]] = list
    # print("分组后的情况：",groupDict)
    return userTaskSize, userBid


# 就算某个任务集合的总体价值；
def setValueCompute(taskSet, set):
    value = 0
    if (len(set) == 0):
        return 0
    else:
        for item in set:
            value = value + taskSet[item]
        return value


# 得到每个user的任务集合
def getUserTaskSet(user, userTaskSet, totalTaskNum):
    userSet = set()
    for i in range(totalTaskNum):
        if (userTaskSet[i][user] == 1):
            userSet.add(i)
    return userSet


def pricePerMarginalValue(user, R, userBid, taskSet, userTaskSet, totalTaskNum):
    user_bid = userBid[user]

    # 得到每个user的任务集合
    user_set = getUserTaskSet(user, userTaskSet, totalTaskNum)
    # 计算user的边际价值
    user_marginal_set = user_set - R
    user_marginal_value = 0
    if len(user_marginal_set) != 0:
        for item in user_marginal_set:
            user_marginal_value = user_marginal_value + taskSet[item]
        pricePerValue = round(user_bid / user_marginal_value)
    else:
        pricePerValue = math.inf
    return pricePerValue, user_marginal_value, user_bid, user_set


def Argmin(groupList, R, userBid, taskSet, userTaskSet, totalTaskNum):
    # 依次遍历list中所有的用户找出最小的性价比的用户；
    minPricePerValue = math.inf
    minMarginalValue = 0
    minUserBid = -1
    minUser = -1
    minUserSet = set()
    for user in groupList:
        pricePerValue, marginalValue, user_bid, user_set = pricePerMarginalValue(user, R, userBid, taskSet, userTaskSet,
                                                                                 totalTaskNum)
        if (marginalValue > 0 and pricePerValue < minPricePerValue):
            minPricePerValue = pricePerValue
            minMarginalValue = marginalValue
            minUserBid = user_bid
            minUserSet = user_set
            minUser = user
    return minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid


def GreedyAlgSM(B, taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum):
    R = set()
    S_w = set()
    userPayment = 0
    averageUtility = 0
    temp_list = [i for i in range(totalUserNum)]

    # 处理user相关的信息
    userTaskSize, userBid = userProcess(userCost, userTaskSet, totalUserNum)

    # 选出当前性价比最高的user；
    minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid, taskSet,
                                                                                 userTaskSet, totalTaskNum)
    # 判断当前是否已经已经没有可以选择的user
    if minMarginalValue == 0:
        totalValue = setValueCompute(taskSet, R)
        return userPayment, totalValue, S_w, round(averageUtility / totalUserNum)

    # 选择user i 后的任务集合
    RcupT_i_set = R | minUserSet
    # 选择user i 后的任务集合价值
    # tempR_value = setValueCompute(taskSet, RcupT_i_set)

    # 循环遍历所有的当前组中的所有user
    while (userPayment + minUserBid <= B):
        temp_list.remove(minUser)
        R = RcupT_i_set
        S_w.add(minUser)
        userPayment = userPayment + minUserBid
        averageUtility = averageUtility + minUserBid - userCost[minUser] * userTaskSize[minUser]
        minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid, taskSet,
                                                                                     userTaskSet, totalTaskNum)
        if minMarginalValue == 0:
            totalValue = setValueCompute(taskSet, R)
            return userPayment, totalValue, S_w, round(averageUtility / totalUserNum)
        RcupT_i_set = R | minUserSet
        # tempR_value = setValueCompute(taskSet, RcupT_i_set)
    totalValue = setValueCompute(taskSet, R)
    # 返回更新后的R,S_w,q
    return userPayment, totalValue, S_w, round(averageUtility / totalUserNum)


if __name__ == '__main__':
    budget = 120
    totalTaskNum = 20
    taskValueDis = 5
    totalUserNum = 200
    userCosPerValueDis = 10
    userTaskNumDis = 5
    # budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis = InitialSetting(20, 20, 30,10, 2.5, 4)

    Data = dp.DataGenerate(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    # taskSet = TaskSet(totalTaskNum, taskValueDis)
    taskSet = Data.TaskSet()
    # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
    userTaskSet, userCost = Data.UserTaskSet()
    # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)
    userPayment, finalValue, S_w, averageUtility = GreedyAlgSM(budget, taskSet, userCost, userTaskSet, totalTaskNum,
                                                               totalUserNum)

    # print("taskSet:", taskSet, "\n")
    # print("userTaskSet:", userTaskSet, "\n")
    # print("userCost:", userCost, "\n")
    print("Winner:", S_w, "\n")
    print("Total value:", finalValue)
    print("Payment", round(userPayment))
    print("averageUtility", averageUtility)
