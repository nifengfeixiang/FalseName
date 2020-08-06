import DataPre as dp
import MultiMinded as MM
import SingleMinded as SM
import GreedyMech as GM
import WPG_infocom as wpg
import numpy as np
import copy
import matplotlib.pyplot as plt
import pylab as pl
import math
import time
import random
from random import sample
from scipy.stats import uniform

def computePoint(x_1, y_1, x_2, y_2, x):
    y = (y_2 - y_1) * (x - x_1) / (x_2 - x_1) + y_1
    return y


def getValuePaymentRelation(value_SM, payment_SM, maxValue, indexValue):
    value = np.array([])
    payment = np.array([])
    index = 0
    length = np.shape(value_SM)[0]
    minIndex, maxIndex = -1, 0
    while index < int(maxValue / indexValue):
        tempValue = index * indexValue + indexValue
        if maxIndex == length - 1:
            minIndex = maxIndex - 1
            flag = computePoint(value_SM[minIndex], payment_SM[minIndex], value_SM[maxIndex],
                                payment_SM[maxIndex], tempValue)
            value = np.append(value, np.array([tempValue]))
            payment = np.append(payment, np.array([flag]))
            index = index + 1
        else:
            if value_SM[maxIndex] < tempValue:
                minIndex = maxIndex
                maxIndex = maxIndex + 1
            else:
                if minIndex == -1:
                    flag = computePoint(0, 0, value_SM[maxIndex], payment_SM[maxIndex], tempValue)
                    value = np.append(value, np.array([tempValue]))
                    payment = np.append(payment, np.array([flag]))
                    index = index + 1
                else:
                    flag = computePoint(value_SM[minIndex], payment_SM[minIndex], value_SM[maxIndex],
                                        payment_SM[maxIndex],
                                        tempValue)
                    value = np.append(value, np.array([tempValue]))
                    payment = np.append(payment, np.array([flag]))
                    index = index + 1
    print("value-payment", value, payment)
    return value, payment


def controlUser(budget, taskSet, userCost, userTaskSet, totalTaskNum, indexUserNum, userTaskNumDis, userSetDict,
                userSetSubsetDict):
    initUser = 50
    x_1 = np.array([])
    y_1 = np.array([])
    y_2 = np.array([])
    y_3 = np.array([])
    Y_1 = np.array([])
    Y_2 = np.array([])
    # 从50到800
    for i in range(indexUserNum):
        num = 50 + i * initUser
        print("user 数量为：:", num)
        userCostTemp_1 = userCost[:num]
        userTaskSetTemp_1 = userTaskSet

        userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM,value,payment = SM.SingleMindedAlg(budget / 2, taskSet,
                                                                                      userCostTemp_1,
                                                                                      userTaskSetTemp_1,
                                                                                      totalTaskNum, num,
                                                                                      userTaskNumDis)
        print("SM总价值：", finalValue_SM, "SM平均收益：", averageUtility_SM)
        # userCostTemp_2 = userCost[:num]
        # userTaskSetTemp_2 = userTaskSet[:, :num]
        userPayment_MM, finalValue_MM, S_w_MM, averageUtility_MM,Value_MM,Payment_MM,Value_SPIM_MM,Payment_SPIM_MM = MM.MultiMindedAlg(budget, taskSet,userTaskSet,totalTaskNum, userCost, num,
                                                                                     userSetDict, userSetSubsetDict)
        print("MM总价值：", finalValue_MM, "MM平均收益：", averageUtility_MM)

        userCostTemp_3 = userCost[:num]
        userTaskSetTemp_3 = userTaskSet[:, :num]
        userPayment_GM, finalValue_GM, S_w_GM, averageUtility_GM = GM.GreedyAlgSM(budget, taskSet, userCostTemp_3,
                                                                                  userTaskSetTemp_3, totalTaskNum,
                                                                                  num)
        print("GM总价值：", finalValue_GM, "GM平均收益：", averageUtility_GM, "\n")

        x_1 = np.append(x_1, np.array([num]))
        y_1 = np.append(y_1, np.array([finalValue_SM]))
        y_2 = np.append(y_2, np.array([finalValue_MM]))
        y_3 = np.append(y_3, np.array([finalValue_GM]))
        Y_1 = np.append(Y_1, np.array([averageUtility_SM]))
        Y_2 = np.append(Y_2, np.array([averageUtility_MM]))
    return x_1, y_1, y_2, y_3, Y_1, Y_2


def doControlUser(reNum, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    # user 考虑的组数(user 考虑100-300，每次增加20)
    indexUserNum = int(((totalUserNum - 50) / 50) + 1)
    SM_platformUtility_1, SM_platformUtility_2, MM_platformUtility_2, GM_platformUtility_2, \
    SM_averageUtility_2, MM_averageUtility_2 = np.zeros((indexUserNum,),
                                                        dtype=np.float), np.zeros(
        (indexUserNum,), dtype=np.float), np.zeros((indexUserNum,), dtype=np.float), np.zeros((indexUserNum,),
                                                                                              dtype=np.float), np.zeros(
        (indexUserNum,), dtype=np.float), np.zeros((indexUserNum,),
                                                   dtype=np.float)
    # 总共执行reNum组随机数据然后去平均值画图
    for i in range(reNum):
        print("重复组数：",i,"\n")
        Data = dp.DataGenerate(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
        # taskSet = TaskSet(totalTaskNum, taskValueDis)
        taskSet = Data.TaskSet()
        # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
        userTaskSet, userCost = Data.UserTaskSet()
        # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)

        userSetDict = Data.userSetDictCompute(userTaskSet)
        userSetSubsetDict = Data.userSetSubsetDictCompute(userSetDict)
        # 控制user数量的图
        SM_pu_1, SM_pu_2, MM_pu_2, GM_pu_2, SM_au_2, MM_au_2 = controlUser(budget,
                                                                           taskSet,
                                                                           userCost,
                                                                           userTaskSet,
                                                                           totalTaskNum,
                                                                           indexUserNum,
                                                                           userTaskNumDis, userSetDict,
                                                                           userSetSubsetDict)

        # print("GM_pu_1", GM_pu_1, "\n")
        # print("GM_platformUtility_1", GM_platformUtility_1, "\n")

        SM_platformUtility_1 = SM_platformUtility_1 + SM_pu_1
        SM_platformUtility_2 = SM_platformUtility_2 + SM_pu_2
        MM_platformUtility_2 = MM_platformUtility_2 + MM_pu_2
        GM_platformUtility_2 = GM_platformUtility_2 + GM_pu_2
        SM_averageUtility_2 = SM_averageUtility_2 + SM_au_2
        MM_averageUtility_2 = MM_averageUtility_2 + MM_au_2

        # print(SM_platformUtility_1,SM_platformUtility_2,MM_platformUtility_1,MM_platformUtility_2,SM_averageUtility_1,SM_averageUtility_2,MM_averageUtility_1,MM_averageUtility_2)
    # for i in range(indexUserNum):
    #     SM_platformUtility_2[i]=math.log(SM_platformUtility_2[i]/reNum,10)
    #     MM_platformUtility_2[i] = math.log(MM_platformUtility_2[i] / reNum, 10)
    #     GM_platformUtility_2[i] = math.log(GM_platformUtility_2[i] / reNum, 10)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    # 画图-----------------------------platformUtility 柱状图；


    bar_width = 0.25  # 条形宽度
    index_GM = np.arange(len(SM_platformUtility_1))  # 男生条形图的横坐标
    index_MM = index_GM + bar_width  # 女生条形图的横坐标
    index_SM = index_MM + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_GM, height=GM_platformUtility_2 / reNum, width=bar_width, color='#FF6600', label='GM')
    plt.bar(index_MM, height=MM_platformUtility_2 / reNum, width=bar_width, color='#339966', label='SPBF-MM')
    plt.bar(index_SM, height=SM_platformUtility_2 / reNum, width=bar_width, color='#3366FF', label='SPBF-SM')

    plt.legend()  # 显示图例
    plt.xticks(index_GM + bar_width, SM_platformUtility_1/reNum)
    plt.xlabel('Number of users',font2)  # make axis labels
    plt.ylabel('Platform Utility',font2)  # 纵坐标轴标题
    plt.title('Impact of users',font2)  # 图形标题
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/platformUtility_users_bar" + string + ".pdf")
    plt.show()



    # ----------------画图-platformUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_platformUtility_2 / reNum, 'r', marker='x',
             label='SPBF-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_platformUtility_2 / reNum, 'g', marker='.', label='SPBF-MM')
    plt.plot(SM_platformUtility_1 / reNum, GM_platformUtility_2 / reNum, 'b', marker='*', label='GM')

    plt.title('Impact of users',font2)  # give plot a title
    plt.xlabel('Number of users',font2)  # make axis labels
    plt.ylabel('Platform Utility',font2)

    plt.legend()
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/platformUtility_users" + string + ".pdf")
    plt.show()  # show the plot on the screen



    #-------------------------画图average-utility -bar
    bar_width = 0.25  # 条形宽度
    index_ave_SM = np.arange(len(SM_platformUtility_1))  # 男生条形图的横坐标
    index_ave_MM = index_ave_SM + bar_width  # 女生条形图的横坐标


    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_ave_SM, height=SM_averageUtility_2 / reNum, width=bar_width, color='#3366FF', label='SPBF-SM')
    plt.bar(index_ave_MM, height=MM_averageUtility_2 / reNum, width=bar_width, color='#339966', label='SPBF-MM')

    plt.legend()  # 显示图例
    plt.xticks(index_ave_SM + bar_width, SM_platformUtility_1 / reNum)
    plt.title('Impact of users',font2)  # give plot a title
    plt.xlabel('Number of users',font2)  # make axis labels
    plt.ylabel('Average Utility',font2)
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/avgUtility_users_bar" + string + ".pdf")
    plt.show()

    # 画图-averageUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_averageUtility_2 / reNum, 'r', marker='x',
             label='SPIM-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_averageUtility_2 / reNum, 'g', marker='.', label='SPIM-MM')

    plt.title('Impact of users',font2)  # give plot a title
    plt.xlabel('Number of users',font2)  # make axis labels
    plt.ylabel('Average Utility',font2)

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/avgUtility_users" + string + ".pdf")
    plt.show()  # show the plot on the screen


def controlTask(budget, indexUserNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    initTask = 20
    x_1 = np.array([])
    y_1 = np.array([])
    y_2 = np.array([])
    y_3 = np.array([])
    y_4 = np.array([])
    y_5 = np.array([])
    Y_1 = np.array([])
    Y_2 = np.array([])
    Y_3 = np.array([])
    Y_4 = np.array([])
    # 从20到60
    for i in range(indexUserNum):
        totalTaskNum = 120 + i * initTask
        print("task 数量为：:", totalTaskNum)

        Data = dp.DataGenerate(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
        # taskSet = TaskSet(totalTaskNum, taskValueDis)
        taskSet = Data.TaskSet()
        # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
        userTaskSet, userCost = Data.UserTaskSet()
        # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)

        userSetDict = Data.userSetDictCompute(userTaskSet)
        userSetSubsetDict = Data.userSetSubsetDictCompute(userSetDict)

        userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM,value,payment = SM.SingleMindedAlg(budget / 2, taskSet,
                                                                                      userCost,
                                                                                      userTaskSet,
                                                                                      totalTaskNum, totalUserNum,
                                                                                      userTaskNumDis)
        print("SM总价值：", finalValue_SM, "SM平均收益：", averageUtility_SM)

        userPayment_MM, finalValue_MM,finalValue_SPIM_MM, S_w_MM, averageUtility_MM,averageUtility_SPIM_MM,value1,payemnt1,value2,payment2 = MM.MultiMindedAlg(budget, taskSet,userTaskSet,totalTaskNum, userCost,
                                                                                     totalUserNum,
                                                                                     userSetDict, userSetSubsetDict)
        print("MM总价值：", finalValue_MM, "MM平均收益：", averageUtility_MM)
        print("SPIM_MM总价值：", finalValue_SPIM_MM, "SPIM_MM平均收益：", averageUtility_SPIM_MM)

        userPayment_GM, finalValue_GM, S_w_GM, averageUtility_GM = GM.GreedyAlgSM(budget, taskSet, userCost,
                                                                                  userTaskSet, totalTaskNum,
                                                                                  totalUserNum)
        print("GM总价值：", finalValue_GM, "GM平均收益：", averageUtility_GM)

        userPayment_SPIM_SM, finalValue_SPIM_SM, S_w_SPIM_SM, averageUtility_SPIM_SM, value_SPIM_SM, payment_SPIM_MM = wpg.SybilAlg(budget, taskSet, userCost, userTaskSet,
                                                                                totalTaskNum, totalUserNum,
                                                                                userTaskNumDis)
        print("SPIM_SM总价值：", finalValue_SPIM_SM, "SPIM_SM平均收益：", averageUtility_SPIM_SM,"\n")

        x_1 = np.append(x_1, np.array([totalTaskNum]))
        y_1 = np.append(y_1, np.array([finalValue_SM]))
        y_2 = np.append(y_2, np.array([finalValue_MM]))
        y_3 = np.append(y_3, np.array([finalValue_GM]))
        y_4 = np.append(y_4, np.array([finalValue_SPIM_SM]))
        y_5 = np.append(y_5, np.array([finalValue_SPIM_MM]))

        Y_1 = np.append(Y_1, np.array([averageUtility_SM]))
        Y_2 = np.append(Y_2, np.array([averageUtility_MM]))
        Y_3 = np.append(Y_3, np.array([averageUtility_SPIM_SM]))
        Y_4 = np.append(Y_4, np.array([averageUtility_SPIM_MM]))
    return x_1, y_1, y_2, y_3,y_4,y_5 ,Y_1,Y_2,Y_3,Y_4


def doControlTask(reNum, budget, maxTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    # 初始设置参数
    # task 考虑的组数(user 考虑40-100，每次增加10)
    indexTaskNum = int(((maxTaskNum - 120) / 20) + 1)
    SM_platformUtility_1, SM_platformUtility_2, MM_platformUtility_2, GM_platformUtility_2, \
    SM_averageUtility_2, MM_averageUtility_2,SPIM_SM_platformUtility_2,SPIM_MM_platformUtility_2,SPIM_SM_averageUtility_2,SPIM_MM_averageUtility_2 = np.zeros((indexTaskNum,),
                                                        dtype=np.float), np.zeros(
        (indexTaskNum,), dtype=np.float), np.zeros((indexTaskNum,), dtype=np.float), np.zeros((indexTaskNum,),
                                                                                              dtype=np.float), np.zeros(
        (indexTaskNum,), dtype=np.float), np.zeros((indexTaskNum,),
                                                   dtype=np.float),np.zeros((indexTaskNum,), dtype=np.float),np.zeros((indexTaskNum,), dtype=np.float),np.zeros((indexTaskNum,), dtype=np.float),np.zeros((indexTaskNum,), dtype=np.float)
    # 执行并画图
    for i in range(reNum):
        # 控制user数量的图
        SM_pu_1, SM_pu_2, MM_pu_2, GM_pu_2,SPIM_SM_pu_2,SPIM_MM_pu_2, SM_au_2, MM_au_2,SPIM_SM_au_2,SPIM_MM_au_2 = controlTask(budget, indexTaskNum, taskValueDis,
                                                                           totalUserNum, userCosPerValueDis,
                                                                           userTaskNumDis)

        # print("GM_pu_1", GM_pu_1, "\n")
        # print("GM_platformUtility_1", GM_platformUtility_1, "\n")

        SM_platformUtility_1 = SM_platformUtility_1 + SM_pu_1

        SM_platformUtility_2 = SM_platformUtility_2 + SM_pu_2
        MM_platformUtility_2 = MM_platformUtility_2 + MM_pu_2
        GM_platformUtility_2 = GM_platformUtility_2 + GM_pu_2
        SPIM_SM_platformUtility_2 = SPIM_SM_platformUtility_2 + SPIM_SM_pu_2
        SPIM_MM_platformUtility_2 = SPIM_MM_platformUtility_2 + SPIM_MM_pu_2

        SM_averageUtility_2 = SM_averageUtility_2 + SM_au_2
        MM_averageUtility_2 = MM_averageUtility_2 + MM_au_2
        SPIM_SM_averageUtility_2 =  SPIM_SM_averageUtility_2 + SPIM_SM_au_2
        SPIM_MM_averageUtility_2 = SPIM_MM_averageUtility_2 + SPIM_MM_au_2

        # print(SM_platformUtility_1,SM_platformUtility_2,MM_platformUtility_1,MM_platformUtility_2,SM_averageUtility_1,SM_averageUtility_2,MM_averageUtility_1,MM_averageUtility_2)

    # 画图-----------------------------platformUtility 柱状图；
    bar_width = 0.15  # 条形宽度
    index_GM = np.arange(len(SM_platformUtility_1))  # 男生条形图的横坐标
    index_MM = index_GM + bar_width  # 女生条形图的横坐标
    index_SM = index_MM + bar_width  # 女生条形图的横坐标
    index_SPIM_MM=index_SM+bar_width
    index_SPIM_SM = index_SPIM_MM + bar_width
    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_GM, height=GM_platformUtility_2 / reNum, width=bar_width, color='#FF6600', label='GM')
    plt.bar(index_MM, height=MM_platformUtility_2 / reNum, width=bar_width, color='#339966', label='TBS-MM')
    plt.bar(index_SM, height=SM_platformUtility_2 / reNum, width=bar_width, color='#3366FF', label='TBS-SM')
    plt.bar(index_SPIM_SM, height=SPIM_SM_platformUtility_2 / reNum, width=bar_width, color='#55A868', label='SPIM-S')
    plt.bar(index_SPIM_MM, height=SPIM_MM_platformUtility_2 / reNum, width=bar_width, color='#4C72B0', label='SPIM-M')
    plt.legend()  # 显示图例
    plt.xticks(index_GM + 2*bar_width,SM_platformUtility_1 / reNum)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    plt.title('Impact of Tasks',font2)  # give plot a title
    plt.xlabel('Number of Tasks',font2)  # make axis labels
    plt.ylabel('Platform Utility',font2)  # 图形标题
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/platformUtility_tasks_bar" + string + ".pdf")
    plt.show()

    # 画图-platformUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_platformUtility_2 / reNum, 'r', marker='x',
             label='TBS-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_platformUtility_2 / reNum, 'g', marker='.', label='TBS-MM')
    plt.plot(SM_platformUtility_1 / reNum, GM_platformUtility_2 / reNum, 'b', marker='*', label='GM')
    plt.plot(SM_platformUtility_1 / reNum, SPIM_SM_platformUtility_2 / reNum, 'y', marker='o', label='SPIM-S')
    plt.plot(SM_platformUtility_1 / reNum, SPIM_MM_platformUtility_2 / reNum, 'grey', marker=',', label='SPIM-M')

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    plt.title('Impact of Tasks',font2)  # give plot a title
    plt.xlabel('Number of Tasks',font2)  # make axis labels
    plt.ylabel('Platform Utility',font2)

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("F:/Results/platformUtility_tasks.pdf")
    plt.show()  # show the plot on the screen

    # -------------------------画图average-utility -bar
    bar_width = 0.15  # 条形宽度
    index_ave_SM = np.arange(len(SM_platformUtility_1))  # 男生条形图的横坐标
    index_ave_MM = index_ave_SM + bar_width  # 女生条形图的横坐标
    index_ave_SM = index_ave_MM + bar_width  # 女生条形图的横坐标
    index_ave_SPIM_MM = index_ave_SM + bar_width
    index_ave_SPIM_SM = index_ave_SPIM_MM + bar_width

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_ave_SM, height=SM_averageUtility_2 / reNum, width=bar_width, color='#3366FF', label='TBS-SM')
    plt.bar(index_ave_MM, height=MM_averageUtility_2 / reNum, width=bar_width, color='#339966', label='TBS-MM')
    plt.bar(index_ave_SPIM_SM, height=SPIM_SM_averageUtility_2 / reNum, width=bar_width, color='#55A868', label='SPIM-S')
    plt.bar(index_ave_SPIM_MM, height=SPIM_MM_averageUtility_2 / reNum, width=bar_width, color='#4C72B0', label='SPIM-M')

    plt.legend()  # 显示图例
    plt.xticks(index_ave_SM + 2*bar_width, SM_platformUtility_1 / reNum)
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    plt.title('Impact of Tasks',font2)  # give plot a title
    plt.xlabel('Number of Tasks',font2)  # make axis labels
    plt.ylabel('Average Utility',font2)
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/avgUtility_tasks_bar" + string + ".pdf")
    plt.show()


    # 画图-averageUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_averageUtility_2 / reNum, 'r', marker='x',
             label='TBS-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_averageUtility_2 / reNum, 'g', marker='.', label='TBS-MM')
    plt.plot(SM_platformUtility_1 / reNum, SPIM_SM_averageUtility_2 / reNum, 'y', marker='o', label='SPIM-S')
    plt.plot(SM_platformUtility_1 / reNum, SPIM_MM_averageUtility_2 / reNum, 'grey', marker=',', label='SPIM-M')

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    plt.title('Impact of Tasks',font2)  # give plot a title
    plt.xlabel('Number of Tasks',font2)  # make axis labels
    plt.ylabel('Average Utility',font2)

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("F:/Results/avgUtility_tasks.pdf")
    plt.show()  # show the plot on the screen


def controlBudget(indexBudget, taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum, userTaskNumDis, userSetDict,
                  userSetSubsetDict):
    initUser = 40
    x_1 = np.array([])
    y_1 = np.array([])
    y_2 = np.array([])
    y_3 = np.array([])
    Y_1 = np.array([])
    Y_2 = np.array([])
    # 从50到800
    for i in range(indexBudget):
        budget = 120 + i * initUser
        print("budget为:", budget)

        userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM,value,payment = SM.SingleMindedAlg(budget / 2, taskSet,
                                                                                      userCost,
                                                                                      userTaskSet,
                                                                                      totalTaskNum, totalUserNum,
                                                                                      userTaskNumDis)
        print("SM总价值：", finalValue_SM, "SM平均收益：", averageUtility_SM)
        # userCostTemp_2 = userCost[:num]
        # userTaskSetTemp_2 = userTaskSet[:, :num]
        userPayment_MM, finalValue_MM, S_w_MM, averageUtility_MM,value1,payment1,value2,payment2 = MM.MultiMindedAlg(budget, taskSet, userTaskSet,totalTaskNum,userCost,
                                                                                     totalUserNum,
                                                                                     userSetDict, userSetSubsetDict)
        print("MM总价值：", finalValue_MM, "MM平均收益：", averageUtility_MM)

        userPayment_GM, finalValue_GM, S_w_GM, averageUtility_GM = GM.GreedyAlgSM(budget, taskSet, userCost,
                                                                                  userTaskSet, totalTaskNum,
                                                                                  totalUserNum)
        print("GM总价值：", finalValue_GM, "GM平均收益：", averageUtility_GM, "\n")

        x_1 = np.append(x_1, np.array([budget]))
        y_1 = np.append(y_1, np.array([finalValue_SM]))
        y_2 = np.append(y_2, np.array([finalValue_MM]))
        y_3 = np.append(y_3, np.array([finalValue_GM]))
        Y_1 = np.append(Y_1, np.array([averageUtility_SM]))
        Y_2 = np.append(Y_2, np.array([averageUtility_MM]))
    return x_1, y_1, y_2, y_3, Y_1, Y_2


def doControlBudget(reNum, maxBudget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    # user 考虑的组数(user 考虑100-300，每次增加20)
    indexBudget = int(((maxBudget - 120) / 40) + 1)
    SM_platformUtility_1, SM_platformUtility_2, MM_platformUtility_2, GM_platformUtility_2, \
    SM_averageUtility_2, MM_averageUtility_2 = np.zeros((indexBudget,),
                                                        dtype=np.float), np.zeros(
        (indexBudget,), dtype=np.float), np.zeros((indexBudget,), dtype=np.float), np.zeros((indexBudget,),
                                                                                            dtype=np.float), np.zeros(
        (indexBudget,), dtype=np.float), np.zeros((indexBudget,),
                                                  dtype=np.float)
    # 总共执行reNum组随机数据然后去平均值画图
    for i in range(reNum):
        print("------------------重复次数为-----------：", i, "\n")
        Data = dp.DataGenerate(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
        # taskSet = TaskSet(totalTaskNum, taskValueDis)
        taskSet = Data.TaskSet()
        # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
        userTaskSet, userCost = Data.UserTaskSet()
        # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)

        userSetDict = Data.userSetDictCompute(userTaskSet)
        userSetSubsetDict = Data.userSetSubsetDictCompute(userSetDict)
        # 控制user数量的图
        SM_pu_1, SM_pu_2, MM_pu_2, GM_pu_2, SM_au_2, MM_au_2 = controlBudget(indexBudget, taskSet, userCost,
                                                                             userTaskSet, totalTaskNum, totalUserNum,
                                                                             userTaskNumDis, userSetDict,
                                                                             userSetSubsetDict)

        # print("GM_pu_1", GM_pu_1, "\n")
        # print("GM_platformUtility_1", GM_platformUtility_1, "\n")

        SM_platformUtility_1 = SM_platformUtility_1 + SM_pu_1
        SM_platformUtility_2 = SM_platformUtility_2 + SM_pu_2
        MM_platformUtility_2 = MM_platformUtility_2 + MM_pu_2
        GM_platformUtility_2 = GM_platformUtility_2 + GM_pu_2
        SM_averageUtility_2 = SM_averageUtility_2 + SM_au_2
        MM_averageUtility_2 = MM_averageUtility_2 + MM_au_2

        # print(SM_platformUtility_1,SM_platformUtility_2,MM_platformUtility_1,MM_platformUtility_2,SM_averageUtility_1,SM_averageUtility_2,MM_averageUtility_1,MM_averageUtility_2)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    # 画图-----------------------------platformUtility 柱状图；
    bar_width = 0.25  # 条形宽度
    index_GM = np.arange(len(SM_platformUtility_1))  # 男生条形图的横坐标
    index_MM = index_GM + bar_width  # 女生条形图的横坐标
    index_SM = index_MM + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_GM, height=GM_platformUtility_2 / reNum, width=bar_width, color='#FF6600', label='GM')
    plt.bar(index_MM, height=MM_platformUtility_2 / reNum, width=bar_width, color='#339966', label='SPBF-MM')
    plt.bar(index_SM, height=SM_platformUtility_2 / reNum, width=bar_width, color='#3366FF', label='SPBF-SM')

    plt.legend()  # 显示图例
    plt.xticks(index_GM + bar_width, SM_platformUtility_1 / reNum)
    plt.title('Impact of budget',font2)  # give plot a title
    plt.xlabel('Budget',font2)  # make axis labels
    plt.ylabel('Platform Utility',font2)
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/platformUtility_budget_bar" + string + ".pdf")
    plt.show()

    # 画图-platformUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_platformUtility_2 / reNum, 'r', marker='x',
             label='SPBF-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_platformUtility_2 / reNum, 'g', marker='.', label='SPBF-MM')
    plt.plot(SM_platformUtility_1 / reNum, GM_platformUtility_2 / reNum, 'b', marker='*', label='GM')

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    plt.title('Impact of budget',font2)  # give plot a title
    plt.xlabel('Budget',font2)  # make axis labels
    plt.ylabel('Platform Utility',font2)

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("F:/Results/platformUtility_budget.pdf")
    plt.show()  # show the plot on the screen

    # -------------------------画图average-utility -bar
    bar_width = 0.25  # 条形宽度
    index_ave_SM = np.arange(len(SM_platformUtility_1))  # 男生条形图的横坐标
    index_ave_MM = index_ave_SM + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_ave_SM, height=SM_averageUtility_2 / reNum, width=bar_width, color='#3366FF', label='SPBF-SM')
    plt.bar(index_ave_MM, height=MM_averageUtility_2 / reNum, width=bar_width, color='#339966', label='SPBF-MM')

    plt.legend()  # 显示图例
    plt.xticks(index_ave_SM + bar_width, SM_platformUtility_1 / reNum)
    plt.title('Impact of budget',font2)  # give plot a title
    plt.xlabel('Budget',font2)  # make axis labels
    plt.ylabel('Average Utility',font2)
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/avgUtility_budget_bar" + string + ".pdf")
    plt.show()

    # 画图-averageUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_averageUtility_2 / reNum, 'r', marker='x',
             label='SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_averageUtility_2 / reNum, 'g', marker='.', label='MM')

    plt.title('Impact of budget',font2)  # give plot a title
    plt.xlabel('Budget',font2)  # make axis labels
    plt.ylabel('Average Utility',font2)

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("F:/Results/avgUtility_budget.pdf")
    plt.show()  # show the plot on the screen


def compareBudget(indexBudget, taskSet, userCost, userTaskSet, totalTaskNum, totalUserNum, userTaskNumDis, userSetDict,
                  userSetSubsetDict):
    initBudget = 20
    x_1 = np.array([])
    y_1 = np.array([])
    y_2 = np.array([])
    y_3 = np.array([])

    # Y_1 = np.array([])
    # Y_2 = np.array([])
    # 从40到800
    budget=0
    for i in range(indexBudget):
        budget = 40 + i * initBudget
        print("budget为:", budget)

        userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM, value_SM, payment_SM = SM.SingleMindedAlg(budget / 2,
                                                                                                            taskSet,
                                                                                                            userCost,
                                                                                                            userTaskSet,
                                                                                                            totalTaskNum,
                                                                                                            totalUserNum,
                                                                                                            userTaskNumDis)
        print("SM总价值：", finalValue_SM, "SM平均收益：", averageUtility_SM)
        # userCostTemp_2 = userCost[:num]
        # userTaskSetTemp_2 = userTaskSet[:, :num]
        userPayment_MM, finalValue_MM,finalValue_SPIM_MM, S_w_MM, averageUtility_MM,averageUtility_SPIM_MM,value_MM,payment_MM,value_SPIM_MM,payment_SPIM_MM = MM.MultiMindedAlg(budget, taskSet,userTaskSet,totalTaskNum, userCost,
                                                                                     totalUserNum,
                                                                                     userSetDict, userSetSubsetDict)
        print("MM总价值：", finalValue_MM, "MM平均收益：", averageUtility_MM)

        userPayment_GM, finalValue_GM, S_w_GM, averageUtility_GM = GM.GreedyAlgSM(budget, taskSet, userCost,
                                                                                  userTaskSet, totalTaskNum,
                                                                                  totalUserNum)
        print("GM总价值：", finalValue_GM, "GM平均收益：", averageUtility_GM, "\n")

        x_1 = np.append(x_1, np.array([budget]))
        y_1 = np.append(y_1, np.array([finalValue_SM]))
        y_2 = np.append(y_2, np.array([finalValue_MM]))
        y_3 = np.append(y_3, np.array([finalValue_GM]))

        # Y_1 = np.append(Y_1, np.array([averageUtility_SM]))
        # Y_2 = np.append(Y_2, np.array([averageUtility_MM]))

    # 计算infocom方法SPIM-S
    userPayment_IN, finalValue_IN, S_w_IN,averageUtility_IN, value_IN, payment_IN = wpg.SybilAlg(budget,taskSet, userCost, userTaskSet,
                                                                               totalTaskNum, totalUserNum,
                                                                               userTaskNumDis)


    return x_1, y_1, y_2, y_3, value_IN, payment_IN


def doCompareBudget(reNum, maxBudget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis,
                    maxValue, indexValue):
    # user 考虑的组数(user 考虑100-300，每次增加20)
    indexBudget = int(((maxBudget*2/3 - 40) / 20) + 1)

    budget, finalValue_SM, finalValue_MM, finalValue_GM, finalValue_SPIM_S = np.zeros((indexBudget,),
                                                                                      dtype=np.float), np.zeros(
        (indexBudget,), dtype=np.float), np.zeros((indexBudget,),
                                                  dtype=np.float), np.zeros(
        (indexBudget,), dtype=np.float), np.zeros((indexBudget,),
                                                  dtype=np.float)
    valueControlIndex = int(maxValue / indexValue)
    controlValue_Index, controlValue_SPIM_S, controlValue_SM_S ,controlValue_SPIM_M,controlValue_SM_M = np.zeros((valueControlIndex,),
                                                                          dtype=np.float), np.zeros(
        (valueControlIndex,), dtype=np.float), np.zeros((valueControlIndex,), dtype=np.float),np.zeros(
        (valueControlIndex,), dtype=np.float),np.zeros(
        (valueControlIndex,), dtype=np.float)
    # 总共执行reNum组随机数据然后去平均值画图
    for i in range(reNum):
        print("------------------重复次数为-----------：", i, "\n")
        Data = dp.DataGenerate(maxBudget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
        # taskSet = TaskSet(totalTaskNum, taskValueDis)
        taskSet = Data.TaskSet()
        # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
        userTaskSet, userCost = Data.UserTaskSet()
        # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)

        userSetDict = Data.userSetDictCompute(userTaskSet)
        userSetSubsetDict = Data.userSetSubsetDictCompute(userSetDict)
        # 控制user数量的图
        currentBudget, y_1, y_2, y_3, value_IN, payment_IN = compareBudget(indexBudget, taskSet, userCost,
                                                                           userTaskSet, totalTaskNum, totalUserNum,
                                                                           userTaskNumDis, userSetDict,
                                                                           userSetSubsetDict)
        s_1, s_2, s_3, s_4, value_SM, payment_SM = SM.SingleMindedAlg(maxBudget / 2,
                                                                      taskSet,
                                                                      userCost,
                                                                      userTaskSet,
                                                                      totalTaskNum,
                                                                      totalUserNum,
                                                                      userTaskNumDis)
        s_5, s_6, s_7, s_8,s_9,s_10, value_MM, payment_MM, value_SPIM_MM, payment_SPIM_MM = MM.MultiMindedAlg(
            maxBudget, taskSet,userTaskSet,totalTaskNum, userCost,
            totalUserNum,
            userSetDict, userSetSubsetDict)
        # print("GM_pu_1", GM_pu_1, "\n")
        # print("GM_platformUtility_1", GM_platformUtility_1, "\n")
        budget = budget + currentBudget
        finalValue_SM = finalValue_SM + y_1
        finalValue_MM = finalValue_MM + y_2
        finalValue_GM = finalValue_GM + y_3

        # 得到WPG在固定payment序列下的value（用的参数IN）
        tempValue = np.zeros((indexBudget,), dtype=np.float)
        for i in range(indexBudget):
            for j in range(payment_IN.shape[0]):
                if payment_IN[j] <= 40 * (i + 1):
                    tempValue[i] = value_IN[j]
        finalValue_SPIM_S = finalValue_SPIM_S + tempValue

        # 得到WPG在固定value序列下的payment
        f_1, f_2 = getValuePaymentRelation(value_IN, payment_IN, maxValue, indexValue)
        controlValue_SPIM_S = controlValue_SPIM_S + f_2

        # 得到WPG_MM在固定value下的payment
        f_1, f_2 = getValuePaymentRelation(value_SPIM_MM, payment_SPIM_MM, maxValue, indexValue)
        controlValue_SPIM_M = controlValue_SPIM_M + f_2

        # contral value 得到SM在固定value下的payment
        f_1, f_2 = getValuePaymentRelation(value_SM, payment_SM, maxValue, indexValue)
        controlValue_Index, controlValue_SM_S = f_1, controlValue_SM_S + f_2

        # 得到本文MM在固定value下的payment
        f_1, f_2 = getValuePaymentRelation(value_MM, payment_MM, maxValue, indexValue)
        controlValue_SM_M = controlValue_SM_M + f_2




        # print(SM_platformUtility_1,SM_platformUtility_2,MM_platformUtility_1,MM_platformUtility_2,SM_averageUtility_1,SM_averageUtility_2,MM_averageUtility_1,MM_averageUtility_2)

    # 画图-platformUtility
    plt.figure()
    plt.plot(budget / reNum, finalValue_SM / reNum, 'r', marker='x',
             label='TBS-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(budget / reNum, finalValue_MM / reNum, 'y', marker='.', label='TBS-MM')
    plt.plot(budget / reNum, finalValue_GM / reNum, 'b', marker='*', label='GM')
    plt.plot(budget / reNum, finalValue_SPIM_S / reNum, 'g', marker='o', label='SPIM-S')

    # 设置输出的图片大小figsize = 11,9figure, ax = plt.subplots(figsize=figsize)
    # #在同一幅图片上画两条折线A,=plt.plot(x1,y1,'-r',label='A',linewidth=5.0)B,=plt.plot(x2,y2,'b-.',label='B',linewidth=5.0)
    # #设置图例并且设置图例的字体及大小font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 23,}
    # legend = plt.legend(handles=[A,B],prop=font1)
    # #设置坐标刻度值的大小以及刻度值的字体plt.tick_params(labelsize=23)labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    # #设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    # plt.title('Impact of budgets', font2)  # give plot a title
    plt.xlabel('Budgets', font2)  # make axis labels
    plt.ylabel('Platform Utility', font2)

    # pl.xlim(40, 2000)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/paymentCompare" + string + ".pdf")
    # plt.savefig("paymentCompare.pdf")
    plt.show()  # show the plot on the screen

    # 画图-control value
    plt.figure()
    plt.plot(controlValue_Index , controlValue_SM_S / reNum, 'r', marker='x',
             label='TBS-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(controlValue_Index, controlValue_SM_M / reNum, 'y', marker='.', label='TBS-MM')
    plt.plot(controlValue_Index , controlValue_SPIM_S / reNum, 'g', marker='.', label='SPIM-S')
    plt.plot(controlValue_Index, controlValue_SPIM_M / reNum, 'b', marker='.', label='SPIM-M')


    # 设置输出的图片大小figsize = 11,9figure, ax = plt.subplots(figsize=figsize)
    # #在同一幅图片上画两条折线A,=plt.plot(x1,y1,'-r',label='A',linewidth=5.0)B,=plt.plot(x2,y2,'b-.',label='B',linewidth=5.0)
    # #设置图例并且设置图例的字体及大小font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 23,}
    # legend = plt.legend(handles=[A,B],prop=font1)
    # #设置坐标刻度值的大小以及刻度值的字体plt.tick_params(labelsize=23)labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    # #设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    # plt.title('Payment Compare', font2)  # give plot a title
    plt.xlabel('Value', font2)  # make axis labels
    plt.ylabel('Payment', font2)

    # pl.xlim(40, 2000)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    string=time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/valuePaymentCompare"+string+".pdf")
    plt.show()  # show the plot on the screen
    print(controlValue_Index,controlValue_SM_S / reNum, controlValue_SPIM_S / reNum,controlValue_SM_M/reNum,controlValue_SPIM_M/reNum)

def truthfulnessValidate(maxBudget,totalTaskNum,taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    Data = dp.DataGenerate(maxBudget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    # taskSet = TaskSet(totalTaskNum, taskValueDis)
    taskSet = Data.TaskSet()
    # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
    userTaskSet, userCost = Data.UserTaskSet()
    # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)

    userSetDict = Data.userSetDictCompute(userTaskSet)
    userSetSubsetDict = Data.userSetSubsetDictCompute(userSetDict)

    # 计算没有谎报情形下的最终结果；
    userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM, value_SM, payment_SM = SM.SingleMindedAlg(budget / 2,
                                                                                                        taskSet,
                                                                                                        userCost,
                                                                                                        userTaskSet,
                                                                                                        totalTaskNum,
                                                                                                        totalUserNum,
                                                                                                      userTaskNumDis)
    userSet=Data.getUserSet(totalUserNum)
    #得到随机的一个loser
    losers=userSet-S_w_SM
    loser=random.sample(losers,1).pop()
    #得到loser的真实utility
    truthful_cost_loser = userCost[loser]
    truthful_utility_loser = userPayment_SM[loser]
    print("loser:",truthful_cost_loser,truthful_utility_loser,"\n")

    selectSet = set(random.sample(S_w_SM, 2))
    print(selectSet)
    selectSetTemp=copy.deepcopy(selectSet)
    #计算两个seller的真实收益
    seller_1 = selectSetTemp.pop()
    truthful_cost_1=userCost[seller_1]
    truthful_utility_1= userPayment_SM[seller_1]-userCost[seller_1]*len(SM.getUserTaskSet(seller_1,userTaskSet,totalTaskNum))
    print("loser:", truthful_cost_1, truthful_utility_1, "\n")
    seller_2 = selectSetTemp.pop()
    truthful_cost_2 = userCost[seller_2]
    truthful_utility_2 = userPayment_SM[seller_2] - userCost[seller_2] * len(
        SM.getUserTaskSet(seller_2, userTaskSet, totalTaskNum))
    print("loser:", truthful_cost_2, truthful_utility_2, "\n")

    #得到测试用的三个user集合
    sellerList=list(copy.deepcopy(selectSet))
    sellerList.append(loser)
    print(sellerList)
    #假定现在每个user在single-minded case可能谎报他的cost
    selectSeller_bid = np.array([1,2,3,4,5,6,7,8,9,10])
    selectSellerBidUtility=np.array([1,2,3,4,5,6,7,8,9,10])

    for item in sellerList:
        print(item)
        selectSeller_utility = np.array([])
        for bid in range(userCosPerValueDis):
            userCost_temp=copy.deepcopy(userCost)
            userCost_temp[item]=bid
            x_0_payment, x_1, S_w,  x_3,  x_4,  x_5 = SM.SingleMindedAlg(
                budget / 2,
                taskSet,
                userCost_temp,
                userTaskSet,
                totalTaskNum,
                totalUserNum,
                userTaskNumDis)
            if item in S_w:
                utility=x_0_payment[item]-len(SM.getUserTaskSet(item,userTaskSet,totalTaskNum))*userCost[item]
            else:
                utility=0
            selectSeller_utility = np.append(selectSeller_utility, np.array([utility]))
        selectSellerBidUtility=np.row_stack((selectSellerBidUtility,selectSeller_utility))
    print(selectSellerBidUtility)

    #画图
    plt.figure()
    plt.plot(selectSellerBidUtility[0], selectSellerBidUtility[1], 'r', marker='x',markersize=8,
             label='user:'+str(sellerList[0])+', cost='+str(round(userCost[sellerList[0]],2)))  # use pylab to plot x and y : Give your plots names
    plt.plot(truthful_cost_1,truthful_utility_1,'r',marker='x',markersize=16)

    plt.plot(selectSellerBidUtility[0], selectSellerBidUtility[2], 'g', marker='.',markersize=8, label='user:'+str(sellerList[1])+', cost='+str(round(userCost[sellerList[1]],2)))
    plt.plot(truthful_cost_2, truthful_utility_2, 'g', marker='.',markersize=16)

    plt.plot(selectSellerBidUtility[0], selectSellerBidUtility[3], 'b', marker='*',markersize=8, label='user:'+str(sellerList[2])+', cost='+str(round(userCost[sellerList[2]],2)))
    plt.plot(truthful_cost_loser,truthful_utility_loser,  'b', marker='*',markersize=16)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    # plt.title('Impact of budget', font2)  # give plot a title
    plt.xlabel('Bids', font2)  # make axis labels
    plt.ylabel('User Utilities', font2)

    # pl.xlim(40, 2000)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/BidUserUtility_SM" + string + ".pdf")
    # plt.savefig("paymentCompare.pdf")
    plt.show()  # show the plot on the screen

def falseName(num,seller,budget,taskSet,userCost,userTaskSet,totalTaskNum,totalUserNum,userCosPerValueDis,userTaskNumDis):
    #更新数据usercost
    userCostTemp=copy.deepcopy(userCost)
    totalUserNumTemp=totalUserNum+num-1
    userCostTemp[seller] = userCost[seller]
    for i in range(num-1):
        userCostTemp = np.append(userCostTemp,np.array(userCost[seller]))

    #更新taskset
    userTaskSetTemp=copy.deepcopy(userTaskSet)
    eachUserTask=SM.getUserTaskSet(seller,userTaskSet,totalTaskNum)
    length=len(eachUserTask)
    R=set()
    randomSize=random.randint(1,length)
    randomSet=set(random.sample(eachUserTask,randomSize))
    for i in range(totalTaskNum):
        if i in randomSet:
            userTaskSetTemp[i][seller]=1
        else:
            userTaskSetTemp[i][seller]=0
    R=R|randomSet
    for i in range(num-1):
        temp=np.array([])
        if i!=num-2:
            randomSize = random.randint(1, length)
            randomSet = set(random.sample(eachUserTask, randomSize))
            for j in range(totalTaskNum):
                if j in randomSet:
                    temp=np.append(temp,np.array([1]))
                else:
                    temp = np.append(temp, np.array([0]))
            R = R | randomSet
        else:
            if R==eachUserTask:
                randomSize = random.randint(1, length)
                randomSet = set(random.sample(eachUserTask, randomSize))
                for j in range(totalTaskNum):
                    if j in randomSet:
                        temp = np.append(temp, np.array([1]))
                    else:
                        temp = np.append(temp, np.array([0]))
            else:
                randomSet=eachUserTask-R
                for j in range(totalTaskNum):
                    if j in randomSet:
                        temp = np.append(temp, np.array([1]))
                    else:
                        temp = np.append(temp, np.array([0]))
        userTaskSetTemp=np.c_[userTaskSetTemp,temp]

    # 计算没有谎报之后下的最终结果；
    userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM, value_SM, payment_SM = SM.SingleMindedAlg(budget / 2,
                                                                                                        taskSet,
                                                                                                        userCostTemp,
                                                                                                        userTaskSetTemp,
                                                                                                        totalTaskNum,
                                                                                                        totalUserNumTemp,
                                                                                                        userTaskNumDis)
    sellerList=[seller]
    for i in range(num-1):
        sellerList.append(totalUserNum+i)

    #计算false之后的收益：
    totalUtility=0
    R_prime=set()
    for item in sellerList:
        if item in S_w_SM:
            tempSet=SM.getUserTaskSet(item,userTaskSetTemp,totalTaskNum)
            R_prime=R_prime| tempSet
            totalUtility = totalUtility + userPayment_SM[item] - len(tempSet) * userCostTemp[item]
    if R_prime!=eachUserTask:
        totalUtility=0
    return totalUtility


def falsenameValidate(maxNum,maxBudget,totalTaskNum,taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    Data = dp.DataGenerate(maxBudget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    # taskSet = TaskSet(totalTaskNum, taskValueDis)
    taskSet = Data.TaskSet()
    # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
    userTaskSet, userCost = Data.UserTaskSet()
    # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)

    userSetDict = Data.userSetDictCompute(userTaskSet)
    userSetSubsetDict = Data.userSetSubsetDictCompute(userSetDict)

    # 计算没有谎报情形下的最终结果；
    userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM, value_SM, payment_SM = SM.SingleMindedAlg(budget / 2,
                                                                                                        taskSet,
                                                                                                        userCost,
                                                                                                        userTaskSet,
                                                                                                        totalTaskNum,
                                                                                                        totalUserNum,
                                                                                                        userTaskNumDis)
    userSet = Data.getUserSet(totalUserNum)
    # 得到随机的一个loser
    losers = userSet - S_w_SM
    loser = random.sample(losers, 1).pop()
    # 得到loser的真实utility
    truthful_cost_loser = userCost[loser]
    truthful_utility_loser = userPayment_SM[loser]
    print("loser:", truthful_cost_loser, truthful_utility_loser, "\n")

    selectSet = set(random.sample(S_w_SM, 2))
    selectSetTemp = copy.deepcopy(selectSet)
    # 计算两个seller的真实收益
    seller_1 = selectSetTemp.pop()
    truthful_cost_1 = userCost[seller_1]
    truthful_utility_1 = userPayment_SM[seller_1] - userCost[seller_1] * len(
        SM.getUserTaskSet(seller_1, userTaskSet, totalTaskNum))
    print("winner:", truthful_cost_1, truthful_utility_1, "\n")
    seller_2 = selectSetTemp.pop()
    truthful_cost_2 = userCost[seller_2]
    truthful_utility_2 = userPayment_SM[seller_2] - userCost[seller_2] * len(
        SM.getUserTaskSet(seller_2, userTaskSet, totalTaskNum))
    print("winner:", truthful_cost_2, truthful_utility_2, "\n")

    # 得到测试用的三个user集合
    sellerList = list(copy.deepcopy(selectSet))
    sellerList.append(loser)
    print(sellerList)


    # 假定现在每个user在single-minded case可能谎报他的cost
    selectSeller_falsename_num = np.array([1, 2, 3, 4, 5])
    selectSellerFalsenameUtility = np.array([1, 2, 3, 4, 5])

    for item in sellerList:
        # 计算这个item在不同false的情况下的收益；
        selectSeller_utility = np.array([])
        for num in range(maxNum-1):
            #根据新的num生成新的数据；
            utility=falseName(num+2,item,budget,taskSet,userCost,userTaskSet,totalTaskNum,totalUserNum,userCosPerValueDis,userTaskNumDis)
            selectSeller_utility = np.append(selectSeller_utility, np.array([utility]))
        print("utitli",selectSeller_utility)
        selectSellerFalsenameUtility = np.row_stack((selectSellerFalsenameUtility, selectSeller_utility))
    print(selectSellerFalsenameUtility)
    selectSellerFalsenameUtility=np.c_[np.array([0,truthful_utility_1,truthful_utility_2,truthful_utility_loser]),selectSellerFalsenameUtility]

    # 画图
    plt.figure()
    plt.plot(selectSellerFalsenameUtility[0], selectSellerFalsenameUtility[1], 'r', marker='x', markersize=8,
             label='user:' + str(sellerList[0]) )  # use pylab to plot x and y : Give your plots names
    plt.plot(0, truthful_utility_1, 'r', marker='x', markersize=16)

    plt.plot(selectSellerFalsenameUtility[0], selectSellerFalsenameUtility[2], 'g', marker='.', markersize=8,
             label='user:' + str(sellerList[1]))
    plt.plot(0, truthful_utility_2, 'g', marker='.', markersize=16)

    plt.plot(selectSellerFalsenameUtility[0], selectSellerFalsenameUtility[3], 'b', marker='*', markersize=8,
             label='user:' + str(sellerList[2]))
    plt.plot(0, truthful_utility_loser, 'b', marker='*', markersize=16)

    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }
    # plt.title('Impact of budget', font2)  # give plot a title
    plt.xlabel('Number of false names', font2)  # make axis labels
    plt.ylabel('User Utilities', font2)

    # pl.xlim(40, 2000)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    string = time.strftime('%Y%m%d%H%M%S')
    plt.savefig("F:/Results/FalsenameUserUtility_SM" + string + ".pdf")
    # plt.savefig("paymentCompare.pdf")
    plt.show()  # show the plot on the screen


if __name__ == '__main__':
    # # 初始设置参数
    # reNum = 100
    # budget = 200
    # totalTaskNum = 150
    # taskValueDis = 20
    # totalUserNum =300
    # userCosPerValueDis = 10
    # userTaskNumDis = 5
    #
    # doControlUser(reNum, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    #
    # # 初始设置参数
    # reNum = 1
    # budget = 200
    # totalTaskNum = 200
    # taskValueDis = 20
    # totalUserNum = 300
    # userCosPerValueDis =10
    # userTaskNumDis = 5
    # maxTaskNum = totalTaskNum
    # doControlTask(reNum, budget, maxTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    #
    # reNum = 100
    # budget = 400
    # totalTaskNum = 150
    # taskValueDis = 20
    # totalUserNum = 200
    # userCosPerValueDis = 10
    # userTaskNumDis = 5
    #
    # doControlBudget(reNum, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)

    # reNum = 20
    # budget = 800
    # totalTaskNum = 150
    # taskValueDis = 5
    # totalUserNum = 200
    # userCosPerValueDis = 10
    # userTaskNumDis = 6

    reNum = 20
    budget = 600
    totalTaskNum = 150
    taskValueDis = 20
    totalUserNum = 200
    userCosPerValueDis = 10
    userTaskNumDis = 5
    maxValue = 700
    indexValue = 50

    doCompareBudget(reNum, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis,
                    maxValue, indexValue)

    # reNum = 10
    # budget = 600
    # totalTaskNum = 150
    # taskValueDis = 20
    # totalUserNum = 200
    # userCosPerValueDis = 10
    # userTaskNumDis = 5
    # maxNum=6
    #
    #
    # # truthfulnessValidate(budget,totalTaskNum,taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
    #
    # falsenameValidate(maxNum, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)