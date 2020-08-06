import DataPre as dp
import numpy as np
import copy
import math
# from numba import jit

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

# # 得到每个user的任务集合的字典
# def userSetDictCompute(userTaskSet, totalUserNum, totalTaskNum):
#     userSetDict = {}
#     for i in range(totalUserNum):
#         tempSet = set()
#         for j in range(totalTaskNum):
#             if (userTaskSet[j][i] == 1):
#                 tempSet.add(j)
#         userSetDict[i] = tempSet
#     return userSetDict
#
#
# # 计算user集合的除去空集的所有子集的字典表示，用list表示所有的子集
# def userSetSubsetDictCompute(userSetDict, totalUserNum):
#     userSetSubsetDict = {}
#     for user in range(totalUserNum):
#         items = list(userSetDict[user])
#         # generate all combination of N items
#         N = len(items)
#         # enumerate the 2**N possible combinations
#         set_all = []
#         for i in range(2 ** N):
#             combo = []
#             for j in range(N):
#                 if (i >> j) % 2 == 1:
#                     combo.append(items[j])
#             set_all.append(combo)
#         userSetSubsetDict[user] = set_all
#     return userSetSubsetDict


# 计算每个user子集的payment，同时确定收益最大的子集
def userPaymentDetermination(taskSet, userCost, totalUserNum, userSetSubsetDict):
    # 初始化A_i,p_i
    userA = {}
    userP = {}
    for user in range(totalUserNum):
        p_i = 0
        utility = 0
        A_i = set()
        subsetList = userSetSubsetDict[user]
        for userItem in subsetList:
            V_item = setValueCompute(taskSet, userItem)
            user_set = set(userItem)
            #得到user某个子集的cost
            userItem_cost = userCost[user] * len(user_set)
            # 遍历其他user
            tempMax = 0
            for otherUser in range(totalUserNum):
                if (otherUser != user):
                    # 遍历这个用户的所有子集；
                    for otherUserItem in userSetSubsetDict[otherUser]:
                        otheruser_set = set(otherUserItem)
                        if (len(user_set & otheruser_set) != 0):
                            tempMax = max(tempMax, setValueCompute(taskSet, otheruser_set) - userCost[otherUser] * len(
                                otheruser_set))
            p_i_userItem = V_item - max(0, tempMax)
            if (p_i_userItem - userItem_cost > utility):
                A_i = user_set
                p_i = p_i_userItem
        userA[user] = A_i
        userP[user] = p_i
    return userA, userP


# Multi-minded 算法主体
def MultiMindedAlg(B, taskSet,userTaskSet,totalTaskNum, userCost,  totalUserNum,userSetDict, userSetSubsetDict):
    # 记录value-payment相关行的行列
    tempValue_MM = np.array([])
    tempPayment_MM = np.array([])
    tempValue_SPIM_MM = np.array([])
    tempPayment_SPIM_MM = np.array([])
    # 计算所有的user的payment
    # userSetDict = userSetDictCompute(userTaskSet, totalUserNum, totalTaskNum)
    # userSetSubsetDict = userSetSubsetDictCompute(userSetDict, totalUserNum)
    userA, userP = userPaymentDetermination(taskSet, userCost, totalUserNum, userSetSubsetDict)

    # ---------首先按照SPIM_MM中的方法进行选择
    totalUtility_SPIM=0
    finalValue_SPIM=0
    temp_R1, temp_payment1 = set(), 0
    for user in range(totalUserNum):
        if userP[user]>0:
            temp_payment1 += userP[user]
            if (temp_payment1)<=B:
                temp_R1 = temp_R1 | userA[user]
                finalValue_SPIM=setValueCompute(taskSet, temp_R1)
                totalUtility_SPIM=totalUtility_SPIM+userP[user]-len(userA[user])*userCost[user]
                # print("test,",totalUtility_SPIM/totalUserNum,"\n")
                tempValue_SPIM_MM = np.append(tempValue_SPIM_MM, np.array([finalValue_SPIM]))
                tempPayment_SPIM_MM = np.append(tempPayment_SPIM_MM, np.array([temp_payment1]))
            else:
                break

    # --------下面从这里开始进行本文章中的方法进行选择，将所有user的备用A_i按照价值排序
    tempTotalValue=0
    userA_iSetValue = {}
    for user in range(totalUserNum):
        task_set = userA[user]
        value = setValueCompute(taskSet, task_set)
        userA_iSetValue[user] = value
        tempTotalValue=tempTotalValue+value

    # 首先将所有的user的A_i task value 排序；
    items = userA_iSetValue.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    A_iValuesortList = [backitems[i][1] for i in range(0, len(backitems))]
    # print("value排序序列：", sortList, "\n")

    # winner selection
    # 首先将所有的有效的user的 task value计算；
    userSetValue={}
    for user in range(totalUserNum):
        task_set = userSetDict[user]
        value = setValueCompute(taskSet, task_set)
        userSetValue[user]=value


    # 首先将所有的user的 task value 排序；
    items = userSetValue.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    sortList = [backitems[i][1] for i in range(0, len(backitems))]
    # print("value排序序列：", sortList, "\n")

    # 按照序列进行选择winner
    R=set()
    totalPayment = 0
    totalUtility=0
    S_w = set()
    for i in sortList:
        if (userP[i] + totalPayment <= B ):
            if(userP[i]!=0):
                S_w.add(i)
                # print("选择winner：",i)
                # print("分配的任务集以及费用：", userA[i],userP[i],"\n")
                totalPayment = totalPayment + userP[i]
                totalUtility=totalUtility+userP[i]-len(userA[i])*userCost[i]
                # print("test-MM,", totalUtility / totalUserNum, "\n")
                R=R|userA[i]
                # print(len(R),"set R:",R)
        else:
            userA[i] = set()
            B = 0
    finalValue_MM=setValueCompute(taskSet,R)
    # print("finalvalue and total value", finalValue,tempTotalValue)

    temp_R, temp_payment = set(), 0
    for user in S_w:
        temp_R = temp_R | userA[user]
        temp_payment = temp_payment + userP[user]
        tempValue_MM = np.append(tempValue_MM, np.array([setValueCompute(taskSet, temp_R)]))
        tempPayment_MM = np.append(tempPayment_MM, np.array([temp_payment]))

    return round(totalPayment,2), finalValue_MM,finalValue_SPIM, S_w, round(totalUtility/totalUserNum,3),round(totalUtility_SPIM/totalUserNum,3),tempValue_MM,tempPayment_MM,tempValue_SPIM_MM,tempPayment_SPIM_MM


if __name__ == '__main__':
    budget = 200
    totalTaskNum = 150
    taskValueDis = 20
    totalUserNum = 200
    userCosPerValueDis = 10
    userTaskNumDis = 5
    # budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis = InitialSetting(20, 20, 30,10, 2.5, 4)

    Data = dp.DataGenerate(budget,totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    # taskSet = TaskSet(totalTaskNum, taskValueDis)
    taskSet = Data.TaskSet()
    # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
    userTaskSet, userCost = Data.UserTaskSet()
    userSetDict= Data.userSetDictCompute(userTaskSet)
    userSetSubsetDict=Data.userSetSubsetDictCompute(userSetDict)

    # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)
    userPayment, finalValue,finalValue_SPIM, S_w ,averageUtility,averageUtility_SPIM,value,payment,value1,payment1= MultiMindedAlg(budget, taskSet,userTaskSet,totalTaskNum, userCost,  totalUserNum,userSetDict, userSetSubsetDict)

    # print("taskSet:", taskSet, "\n")
    # print("userTaskSet:", userTaskSet, "\n")
    # print("userCost:", userCost, "\n")
    print("Winner:", S_w, "\n")
    print("Final value:", finalValue,finalValue_SPIM)
    print("Payment", userPayment)
    print("averageUtility", averageUtility,averageUtility_SPIM)
    # print(value,payment,"\n")
    # print(value1, payment1,"\n")
