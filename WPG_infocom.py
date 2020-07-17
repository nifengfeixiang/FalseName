import DataPre as dp
import numpy as np
import copy
import math
import SingleMinded as sm


def WinnerSelection(R, S_w, groupList, userBid, taskSet, userTaskSet, totalTaskNum):
    # print("winner 选择，当前考虑分组为：", groupList)
    temp_list = groupList

    # 选出当前性价比最高的user；
    minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = sm.Argmin(temp_list, R, userBid, taskSet,
                                                                                    userTaskSet, totalTaskNum)

    # 判断当前是否已经已经没有可以选择的user
    if minMarginalValue == 0:
        return R, S_w
    # 选择user i 后的任务集合
    # 选择user i 后的任务集合价值
    # tempR_value = sm.setValueCompute(taskSet, RcupT_i_set)
    userTaskValue=sm.setValueCompute(taskSet, minUserSet)
    # print("当前的R集合为：",R)
    # print("性价比最高的user为：", minUser,"报价为：",round(userBid[minUser]),"边际价值为：",minMarginalValue,"value R是",tempR_value)
    # print("边际价值单价为：",minPricePerValue,"此时q_max为：",q,"此时B/R为：",round(B / tempR_value,2))
    # 循环遍历所有的当前组中的所有user
    while (minUserBid<= userTaskValue and len(temp_list)!=0):
        temp_list.remove(minUser)
        R = R | minUserSet
        # print("当前被选择winner为：",minUser,"\n")
        S_w.add(minUser)
        minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = sm.Argmin(temp_list, R, userBid, taskSet,
                                                                                     userTaskSet, totalTaskNum)
        if minMarginalValue == 0:
            return R, S_w

        # 选择user i 后的任务集合价值
        # tempR_value = sm.setValueCompute(taskSet, RcupT_i_set)
        userTaskValue = sm.setValueCompute(taskSet, minUserSet)

        # print("当前的R集合为：", R)
        # print("性价比最高的user为：", minUser, "报价为：", round(userBid[minUser]), "边际价值为：", minMarginalValue, "value R是",
        #       tempR_value)
        # print("边际价值单价为：", minPricePerValue, "此时q_max为：", q, "此时B/R为：", round(B / tempR_value,2) )
    # 返回更新后的R,S_w,q
    return R, S_w

def PaymentScheme(R, groupList, userPayment, userBid, taskSet, userTaskSet, totalTaskNum):
    # print("payment 计算，当前考虑分组为：", groupList,"以及winning set：",S_w)
    for i in groupList:
        if (i in groupList):
            # print("当前考虑winning user为：",i)
            # 计算winner i 的payment
            tempR = R
            p_i = 0
            # user i 的任务集合
            T_i = sm.getUserTaskSet(i, userTaskSet, totalTaskNum)
            # 去除user i
            temp_list = copy.deepcopy(groupList)
            temp_list.remove(i)

            # 判断去除user后的list是否为空；
            if (len(temp_list) != 0):
                # 选出当前性价比最高的user；
                minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = sm.Argmin(temp_list, R, userBid,
                                                                                             taskSet,
                                                                                             userTaskSet, totalTaskNum)
                if minMarginalValue == 0:
                    userPayment[i] = sm.setValueCompute(taskSet, T_i - tempR)
                else:
                    # 循环遍历
                    while (len(temp_list)!=0 and len(tempR)!=len(taskSet)):
                        if minUserBid>sm.setValueCompute(taskSet,minUserSet):
                            break
                        v_i_tempR=sm.setValueCompute(taskSet,T_i|tempR)
                        p_i = max(p_i, min(v_i_tempR * min(minPricePerValue, round(B / tempR_value)), v_i_tempR))
                        temp_list.remove(minUser)

                        tempR = tempR | minUserSet
                        tempR_value = sm.setValueCompute(taskSet, RcupT_i_set)

                        minPricePerValue, minMarginalValue, minUser, minUserSet, minUserBid = Argmin(temp_list, R, userBid,
                                                                                                     taskSet,
                                                                                                     userTaskSet,
                                                                                                     totalTaskNum)
                        if minMarginalValue == 0:
                            break
                        RcupT_i_set = tempR | minUserSet
                        tempR_value = setValueCompute(taskSet, RcupT_i_set)
                    p_i = max(p_i, setValueCompute(taskSet, T_i - tempR))
                    userPayment[i] = p_i
            else:
                userPayment[i] = setValueCompute(taskSet, T_i - tempR)
        # print("当前user paymenmt为：", userPayment[i], "\n")
    # print("当前分组payment：",userPayment,"\n")
    return userPayment


def SybilAlg(taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum, userTaskNumDis):
    # 初始化相关参数
    # budget: B
    finalValue = 0
    # 全局被选择的任务集合
    R = set()
    # 全局winner集合
    S_w = set()
    # 全局payment向量
    userPayment = np.zeros((totalUserNum,), dtype=np.float)

    # 将所有的user进行分组，分别得到：分组后的dict；每个user的task数量array；每个user的总报价array；
    print("---开始将user进行分组---", "\n")
    groupDict, userTaskSize, userBid = sm.Group(userCost, userTaskSet, totalUserNum, userTaskNumDis)
    # print("分组结果为：", groupDict,"\n")

    # 从任务数量最高的组进行循环
    for i in range(userTaskNumDis):
        # 选择可能是task size最大的组
        groupList = groupDict[userTaskNumDis - i]
        # print("考虑分组", groupList,"\n")
        # 备份一些不需要立即更新的参数，便于在计算payment使用：
        tempR = copy.deepcopy(R)
        # tempS_w=copy.deepcopy(S_w)
        tempGroupList = copy.deepcopy(groupList)

        length = len(groupList)
        # 判断这个组里有没有user
        if length != 0:
            # 选择winner，传入备用参数，同时更新全局参数
            R, S_w = WinnerSelection(R, S_w, groupList, userBid, taskSet, userTaskSet, totalTaskNum)
            # 计算payment值，使用之前设置的备份参数
            userPayment = PaymentScheme(tempR, tempGroupList, userPayment, userBid, taskSet, userTaskSet,
                                        totalTaskNum)
        groupSet = set(groupList)
        if (not (groupSet.issubset(S_w))):
            break
    # 计算最终buyer的收益
    finalValue = setValueCompute(taskSet, R)
    totalUtility = 0
    for i in range(totalUserNum):
        if (userPayment[i] != 0):
            totalUtility = totalUtility + userPayment[i] - len(getUserTaskSet(i, userTaskSet, totalTaskNum)) * userCost[
                i]

    return userPayment, finalValue, S_w, round(totalUtility / totalUserNum, 2)
