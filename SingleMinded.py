import DataPre as dp
import numpy as np
import copy
import math


def Group(userCost, userTaskSet, totalUserNum, userTaskNumDis):
    """
    计算每个user的task集合的大小，并分组
    :param userTaskSet: user task set
    :return: 分组；及每个user task的size
    """

    # 计算每个user的集合task size
    size = np.zeros((totalUserNum,), dtype=np.int)

    for i in range(totalUserNum):
        size[i] = sum(userTaskSet[:,i])
    userTaskSize = size
    # print("每个user的集合size：",userTaskSize,"\n")
    # 计算每个用户的报价
    userBid = userTaskSize * userCost


    # 对每个user的size进行分组
    disSequence = []
    for i in range(userTaskNumDis+1):
        disSequence.append(userTaskNumDis - i )
    # print("任务集合大小序列为：",disSequence)
    groupDict = dict.fromkeys(disSequence, [])

    for i in range(totalUserNum):
        list = copy.deepcopy(groupDict[size[i]])
        list.append(i)
        groupDict[size[i]] = list
    # print("分组后的情况：",groupDict)
    return groupDict, userTaskSize, userBid

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
    # user_marginal_value = 0
    if len(user_marginal_set) != 0:
        # for item in user_marginal_set:
        #     user_marginal_value = user_marginal_value + taskSet[item]
        user_marginal_value=setValueCompute(taskSet,user_marginal_set)
        pricePerValue = round(user_bid / user_marginal_value,2)
    else:
        user_marginal_value=0
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


def WinnerSelection(B, R, q, groupList, userBid, taskSet, userTaskSet, totalTaskNum):
    print("------winner 选择------------\n 当前考虑分组为：", groupList)
    temp_list = groupList
    S_w=set()
    # 选出当前性价比最高的user；
    minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid, taskSet,
                                                                              userTaskSet, totalTaskNum)
    print("性价比最高的user为：", minUser, "报价为：", round(userBid[minUser],2), "边际价值为：", minPricePerValue, "用户的task集合为：",minUserSet,"当前已经选择的集合数量为", len(R),"\n")

    #判断当前是否已经已经没有可以选择的user
    if minMarginalValue==0:
        print("边际价值为0，无可选择的winner；\n")
        return R, S_w, q
    #计算user的task value
    userValue=setValueCompute(taskSet,minUserSet)
    # 选择user i 后的任务集合
    RcupT_i_set = R | minUserSet
    # 选择user i 后的任务集合价值
    tempR_value = setValueCompute(taskSet, RcupT_i_set)
    print("当前总体budget/value的值为：",tempR_value, round(B / tempR_value, 2), "\n")
    # 进行winner选择，循环遍历所有的当前组中的所有user
    while (minPricePerValue <= (B / tempR_value) and q <= (B / tempR_value) and len(
            temp_list) != 0):
        #tempR_value<=B, minUserBid<=v_{j}条件先不添加
        temp_list.remove(minUser)
        q = max(q, minPricePerValue)
        R = R | minUserSet
        print("当前被选择winner为：",minUser,"\n")
        S_w.add(minUser)

        # 下一轮找winner
        minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid, taskSet,
                                                                                     userTaskSet, totalTaskNum)

        print("性价比最高的user为：", minUser, "报价为：", round(userBid[minUser],2), "边际价值为：", minPricePerValue, "用户的task集合为：",
              minUserSet, "当前已经选择的集合数量为", len(R), "\n")
        if minMarginalValue == 0:
            print("边际价值为0，无可选择的winner；\n")
            return R, S_w, q
        userValue = setValueCompute(taskSet, minUserSet)
        RcupT_i_set = R | minUserSet
        tempR_value = setValueCompute(taskSet, RcupT_i_set)
        print("当前总体budget/value的值为：",tempR_value,round(B/tempR_value,2),"\n")
        # print("当前的R集合为：", R)
        # print("性价比最高的user为：", minUser, "报价为：", round(userBid[minUser]), "边际价值为：", minMarginalValue, "value R是",
        #       tempR_value)
        # print("边际价值单价为：", minPricePerValue, "此时q_max为：", q, "此时B/R为：", round(B / tempR_value,2) )
    # 返回更新后的R,S_w,q
    print("------结束当前组选择winner------------")
    return R, S_w, q


def PaymentScheme(B, R, q, S_w, groupList, userPayment, userBid, taskSet, userTaskSet, totalTaskNum):
    # print("payment 计算，当前考虑分组为：", groupList,"以及winning set：",S_w)
    print("------payment 计算------------\n 当前考虑分组为：", groupList)
    for i in groupList:
        if (i in S_w):
            # print("当前考虑winning user为：",i)
            # 计算winner i 的payment
            tempR = copy.deepcopy(R)
            q_prime = copy.deepcopy(q)
            p_i = 0
            # user i 的任务集合
            T_i = getUserTaskSet(i, userTaskSet, totalTaskNum)
            # 去除user i
            temp_list = copy.deepcopy(groupList)
            temp_list.remove(i)

            # 判断去除user后的list是否为空；
            if (len(temp_list) != 0):
                # 选出当前性价比最高的user；
                minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid,
                                                                                             taskSet,
                                                                                             userTaskSet, totalTaskNum)
                if minMarginalValue == 0:
                    userPayment[i] = setValueCompute(taskSet, T_i - tempR)
                else:
                    userValue = setValueCompute(taskSet, minUserSet)
                    RcupT_i_set = tempR | minUserSet
                    tempR_value = setValueCompute(taskSet, RcupT_i_set)

                    # 循环遍历
                    while (minPricePerValue <= (B / tempR_value) and q_prime <= (B / tempR_value) and len(
                            temp_list) != 0 ):
                        #and  userBid[minUser] <= userValue
                        v_i_tempR = setValueCompute(taskSet, T_i - tempR)
                        p_i = max(p_i, min(v_i_tempR * min(minPricePerValue, round(B / tempR_value)), v_i_tempR))
                        temp_list.remove(minUser)
                        q_prime = max(q_prime, minPricePerValue)
                        tempR = tempR | minUserSet
                        minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid,
                                                                                                     taskSet,
                                                                                                     userTaskSet,
                                                                                                     totalTaskNum)
                        if minMarginalValue == 0:
                            break
                        userValue = setValueCompute(taskSet, minUserSet)
                        RcupT_i_set = tempR | minUserSet
                        tempR_value = setValueCompute(taskSet, RcupT_i_set)

                    p_i = max(p_i, setValueCompute(taskSet, T_i - tempR))
                    userPayment[i] = p_i
            else:
                userPayment[i] = setValueCompute(taskSet, T_i - tempR)
        # print("当前user paymenmt为：", userPayment[i], "\n")
    # print("当前分组payment：",userPayment,"\n")
    return userPayment

def SingleMindedAlg(B, taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum, userTaskNumDis):
    # 初始化相关参数
    # budget: B
    finalValue = 0
    # 全局被选择的任务集合
    R = set()
    # 全局winner集合
    S_w = set()
    # 全局payment向量
    userPayment = np.zeros((totalUserNum,), dtype=np.float)
    # 初始参考用的q值
    q = 0

    # 将所有的user进行分组，分别得到：分组后的dict；每个user的task数量array；每个user的总报价array；
    # print("---开始将user进行分组---","\n")
    groupDict, userTaskSize, userBid = Group(userCost, userTaskSet, totalUserNum, userTaskNumDis)
    # print("分组结果为：", groupDict,"\n")


    # 从任务数量最高的组进行循环
    for i in range(userTaskNumDis):
        # 选择可能是task size最大的组
        print("------------考虑分组----------\n",userTaskNumDis - i)
        groupList = groupDict[userTaskNumDis - i]
        print(groupList, "\n")

        # 备份一些不需要立即更新的参数，便于在计算payment使用：
        tempR = copy.deepcopy(R)
        tempQ = copy.deepcopy(q)
        # tempS_w=copy.deepcopy(S_w)
        tempGroupListForWinner = copy.deepcopy(groupList)
        tempGroupListForPayment = copy.deepcopy(groupList)
        length = len(groupList)
        # 判断这个组里有没有user
        if length != 0:
            # 选择winner，传入备用参数，同时更新全局参数
            R, tempS_w, q = WinnerSelection(B, R, q,  tempGroupListForWinner, userBid, taskSet, userTaskSet, totalTaskNum)
            S_w=S_w|tempS_w
            # 计算payment值，使用之前设置的备份参数
            userPayment=PaymentScheme(B, tempR, tempQ, S_w, tempGroupListForPayment, userPayment, userBid, taskSet, userTaskSet, totalTaskNum)
            groupSet=set(groupList)
            if (not (groupSet.issubset(tempS_w))):
                break
        # 释放深度拷贝的参数

    # 计算最终buyer的收益
    finalValue = setValueCompute(taskSet, R)
    totalUtility=0
    for i in range(totalUserNum):
        if(userPayment[i]!=0):
            totalUtility=totalUtility+userPayment[i]-len(getUserTaskSet(i,userTaskSet,totalTaskNum))*userCost[i]

    return userPayment, finalValue, S_w, round(totalUtility/totalUserNum,2)


if __name__ == '__main__':
    reNum = 1
    budget = 200
    totalTaskNum = 200
    taskValueDis = 5
    totalUserNum = 50
    userCosPerValueDis = 10
    userTaskNumDis = 5

    # budget = 120
    # totalTaskNum = 150
    # taskValueDis = 5
    # totalUserNum = 200
    # userCosPerValueDis = 2.5
    # userTaskNumDis = 5
    # budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis = InitialSetting(20, 20, 30,10, 2.5, 4)

    Data = dp.DataGenerate(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    # taskSet = TaskSet(totalTaskNum, taskValueDis)
    taskSet = Data.TaskSet()
    # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
    userTaskSet, userCost = Data.UserTaskSet()
    # print("usercost",userCost)
    # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)
    userPayment, finalValue, S_w ,averageUtility= SingleMindedAlg(budget, taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum,
                                                   userTaskNumDis)

    totalvalue=setValueCompute(taskSet,taskSet)
    # print("taskSet:", taskSet, "\n")
    # print("userTaskSet:", userTaskSet, "\n")
    # print("userCost:", userCost, "\n")
    print("Winner:", S_w, "\n")
    print("Total value:", finalValue)
    print("Payment list", sum(userPayment))
    print("averageUtility", averageUtility)
    # print("total value",totalvalue)