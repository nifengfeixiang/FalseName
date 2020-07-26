import DataPre as dp
import numpy as np
import copy
import math
import SingleMinded as sm
import matplotlib.pyplot as plt


def pricePerMarginalValue(user, R, userBid, taskSet, userTaskSet, totalTaskNum):
    user_bid = userBid[user]

    # 得到每个user的任务集合
    user_set = sm.getUserTaskSet(user, userTaskSet, totalTaskNum)
    # 计算user的边际价值
    user_marginal_set = user_set - R
    # user_marginal_value = 0
    user_marginal_value = sm.setValueCompute(taskSet, user_marginal_set)
    if user_marginal_value > 0:
        pricePerValue = round(user_bid / user_marginal_value, 5)
    else:
        pricePerValue = 100000
    return pricePerValue, user_marginal_value, user_bid, user_set


def Argmin(groupList, R, userBid, taskSet, userTaskSet, totalTaskNum):
    # 依次遍历list中所有的用户找出最小的性价比的用户；
    minPricePerValue = math.inf
    minMarginalValue = -2
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


def WinnerSelection(R, S_w, winnerSequence, groupList, userBid, taskSet, userTaskSet, totalTaskNum):
    # print("winner 选择，当前考虑分组为：", groupList)
    temp_list = copy.deepcopy(groupList)

    # 选出当前性价比最高的user；
    minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid, taskSet,
                                                                                 userTaskSet, totalTaskNum)

    # 判断当前是否已经已经没有可以选择的user
    # if minMarginalValue == 0:
    #     return R, S_w
    # 选择user i 后的任务集合
    # 选择user i 后的任务集合价值
    # tempR_value = sm.setValueCompute(taskSet, RcupT_i_set)
    # print("当前最高性价比seller是：", minUser)
    userTaskValue = sm.setValueCompute(taskSet, minUserSet)
    # print("当前的R集合为：",R)
    # print("性价比最高的user为：", minUser,"报价为：",round(userBid[minUser]),"边际价值为：",minMarginalValue,"value R是",tempR_value)
    # print("边际价值单价为：",minPricePerValue,"此时q_max为：",q,"此时B/R为：",round(B / tempR_value,2))
    # 循环遍历所有的当前组中的所有user
    while (minUserBid <= minMarginalValue and len(temp_list) != 0):
        temp_list.remove(minUser)
        R = R | minUserSet
        # print("当前被选择winner为：",minUser,"\n")
        S_w.add(minUser)
        winnerSequence.append(minUser)
        minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid, taskSet,
                                                                                     userTaskSet, totalTaskNum)
        # if minMarginalValue == 0:
        #     return R, S_w

        # 选择user i 后的任务集合价值
        # tempR_value = sm.setValueCompute(taskSet, RcupT_i_set)
        userTaskValue = sm.setValueCompute(taskSet, minUserSet)

        # print("当前的R集合为：", R)
        # print("性价比最高的user为：", minUser, "报价为：", round(userBid[minUser], 2), "边际价值为：", minMarginalValue, "\n")
        # print("边际价值单价为：", minPricePerValue, "此时q_max为：", q, "此时B/R为：", round(B / tempR_value,2) )
    # 返回更新后的R,S_w,q
    # print("winner sequence", winnerSequence, "\n")
    return R, S_w, winnerSequence


def PaymentScheme(R, S_w, groupList, userPayment, userBid, taskSet, userTaskSet, totalTaskNum):
    # print("---计算当前组的payment---\n")
    for user in groupList:
        # if (i in groupList):
        # print("当前考虑winning user为：",i)
        # 计算winner i 的payment
        # print("计算user", user, "的payment:")
        if user in S_w:
            tempR = copy.deepcopy(R)

            p_i = 0
            # user i 的任务集合
            T_i = sm.getUserTaskSet(user, userTaskSet, totalTaskNum)
            # print("T_i", T_i, "\n")
            # user i 的报价
            b_i = userBid[user]
            # 去除user i
            temp_list = copy.deepcopy(groupList)
            temp_list.remove(user)
            #去除之后的list为：
            # print("list wei :",temp_list,"\n")
            # 判断去除user后的list是否为空；
            if (len(temp_list) != 0):
                # 选出当前性价比最高的user；
                minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid,
                                                                                             taskSet,
                                                                                             userTaskSet, totalTaskNum)
                # if minMarginalValue == 0:
                #     userPayment[i] = sm.setValueCompute(taskSet, T_i - tempR)
                # else:
                # 循环遍历
                # print("性价比最高的user为：", minUser, "报价为：", round(userBid[minUser], 2), "边际价值为：", minMarginalValue, "\n")
                while (len(temp_list) != 0 and len(tempR) != totalTaskNum):
                    if minUserBid > minMarginalValue:
                        break
                    v_i_tempR = sm.setValueCompute(taskSet, T_i - tempR)
                    p_i = max(p_i, min(v_i_tempR * (minUserBid / minMarginalValue), v_i_tempR))
                    # print("p_i", v_i_tempR,minUserBid / minMarginalValue)

                    # 更新集合
                    tempR = tempR | minUserSet
                    temp_list.remove(minUser)
                    # tempR_value = sm.setValueCompute(taskSet,temv_i_tempRpR)

                    minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid,
                                                                                                 taskSet,
                                                                                                 userTaskSet,
                                                                                                 totalTaskNum)
                    # print("性价比最高的user为：", minUser, "报价为：", round(userBid[minUser], 2), "边际价值为：", minMarginalValue, "\n")
                v_i_tempRPrime = sm.setValueCompute(taskSet, T_i & tempR)
                if b_i <= v_i_tempRPrime:
                    p_i = max(p_i, v_i_tempRPrime)
                userPayment[user] = p_i
                # print(userPayment[i],"\n")
                # else:
                #     userPayment[i] = setValueCompute(taskSet, T_i - tempR)
            # print(userPayment[user], "\n")
    temp = 0
    for item in groupList:
        temp = temp + userPayment[item]
    # print("当前user paymenmt为：", temp, "\n")
    # print("当前分组payment：",userPayment,"\n")
    return userPayment


def SybilAlg(taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum, userTaskNumDis):
    # 初始化相关参数
    # budget: B
    finalValue = 0
    taskNum = len(taskSet)
    # 全局被选择的任务集合
    R = set()
    # 全局winner集合
    S_w = set()
    # 全局payment向量
    userPayment = np.zeros((totalUserNum,), dtype=np.float)
    # 计算当前的payment value的关系；设置两个中间向量temp1和temp2
    temp1, temp2 = 0, 0
    tempR = set()
    # 记录value-payment相关行的行列
    tempValue = np.array([])
    tempPayment = np.array([])
    # 将所有的user进行分组，分别得到：分组后的dict；每个user的task数量array；每个user的总报价array；
    # print("---开始将user进行分组---", "\n")
    groupDict, userTaskSize, userBid = sm.Group(userCost, userTaskSet, totalUserNum, userTaskNumDis)
    # print("分组结果为：", groupDict,"\n")

    # 从任务数量最高的组进行循环
    i = 0
    while i < userTaskNumDis and len(R) != taskNum:
        # 初始化当前winner选择的序列
        winnerSequence = []
        # 选择可能是task size最大的组
        groupList = groupDict[userTaskNumDis - i]
        # print("考虑分组", userTaskNumDis - i,groupList, "\n")
        # 备份一些不需要立即更新的参数，便于在计算payment使用：
        tempR = copy.deepcopy(R)
        # tempS_w=copy.deepcopy(S_w)
        tempGroupList = copy.deepcopy(groupList)

        length = len(groupList)
        # 判断这个组里有没有user
        if length > 0:
            # 选择winner，传入备用参数，同时更新全局参数
            R, S_w, winnerSequence = WinnerSelection(R, S_w, winnerSequence, groupList, userBid, taskSet, userTaskSet,
                                                     totalTaskNum)
            # 计算payment值，使用之前设置的备份参数
            userPayment = PaymentScheme(tempR, S_w,tempGroupList, userPayment, userBid, taskSet, userTaskSet,
                                        totalTaskNum)

        # 计算当前的payment value的关系；设置两个中间向量temp1和temp2
        for user in winnerSequence:
            tempR = tempR | sm.getUserTaskSet(user, userTaskSet, totalTaskNum)
            temp1 = sm.setValueCompute(taskSet, tempR)
            temp2 = temp2 + userPayment[user]

            tempValue = np.append(tempValue, [temp1])
            tempPayment = np.append(tempPayment, [temp2])
        for user in tempGroupList:
            # tempR = tempR | sm.getUserTaskSet(user, userTaskSet, totalTaskNum)
            # temp1 = temp2 + sm.setValueCompute(taskSet, tempR)
            temp2 = temp2 + userPayment[user]
            if user not in winnerSequence:
                tempValue = np.append(tempValue, [temp1])
                tempPayment = np.append(tempPayment, [temp2])
        # print("此次分组的费用", userPayment, "\n")
        i = i + 1
    # 计算最终buyer的收益
    finalValue = sm.setValueCompute(taskSet, R)
    # print("final value:", finalValue,sm.setValueCompute(taskSet, taskSet))
    # totalUtility = 0
    # for i in range(totalUserNum):
    #     if (userPayment[i] != 0):
    #         totalUtility = totalUtility + userPayment[i] - len(getUserTaskSet(i, userTaskSet, totalTaskNum)) * userCost[
    #             i]

    return userPayment, finalValue, S_w,tempValue,tempPayment


if __name__ == '__main__':
    # reNum = 1
    # budget = 300
    # totalTaskNum = 150
    # taskValueDis = 5
    # totalUserNum = 200
    # userCosPerValueDis = 5
    # userTaskNumDis = 5

    reNum = 1
    budget = 600
    totalTaskNum = 150
    taskValueDis = 20
    totalUserNum = 200
    userCosPerValueDis = 10
    userTaskNumDis =5
    # reNum = 20
    # budget = 400
    # totalTaskNum = 150
    # taskValueDis = 5
    # totalUserNum = 200
    # userCosPerValueDis = 10
    # userTaskNumDis = 5

    # budget = 120
    # totalTaskNum = 150
    # taskValueDis = 5
    # totalUserNum = 300
    # userCosPerValueDis = 10
    # userTaskNumDis = 5
    # budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis = InitialSetting(20, 20, 30,10, 2.5, 4)

    Data = dp.DataGenerate(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    # taskSet = TaskSet(totalTaskNum, taskValueDis)
    taskSet = Data.TaskSet()
    # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
    userTaskSet, userCost = Data.UserTaskSet()
    # print("usercost",userCost)
    # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)
    userPayment, finalValue, S_w, value, payment = SybilAlg(taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum,
                                                            userTaskNumDis)

    print("finalvalue", finalValue,"\n")
    print("S_w", S_w, "\n")

    # totalPayment = 0
    # R = set()
    # for winner in winnerSequence:
    #     task = sm.getUserTaskSet(winner, userTaskSet, totalTaskNum)
    #     R = R | task
    #     v = sm.setValueCompute(taskSet, R)
    #     totalPayment = totalPayment + userPayment[winner]
    #     # print("(value,totalpayment:",v,totalPayment)
    #     # 添加此时的value-payment关系组
    #     value = np.append(value, np.array([v]))
    #     payment = np.append(payment, np.array([totalPayment]))

    # 画图-platformUtility
    plt.figure()
    plt.plot(payment, value, 'r', marker='x',
             label='Sybil')  # use pylab to plot x and y : Give your plots names
    # plt.plot(SM_platformUtility_1 / reNum, MM_platformUtility_2 / reNum, 'g', marker='.', label='SPBF-MM')
    # plt.plot(SM_platformUtility_1 / reNum, GM_platformUtility_2 / reNum, 'b', marker='*', label='GM-SM')

    plt.title('Impact of budget')  # give plot a title
    plt.xlabel('payment')  # make axis labels
    plt.ylabel('value')

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("sybil_payment.pdf")
    plt.show()  # show the plot on the screen
