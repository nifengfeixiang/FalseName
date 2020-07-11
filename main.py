import DataPre as dp
import MultiMinded as MM
import SingleMinded as SM
import GreedyMech as GM
import numpy as np
import copy
import matplotlib.pyplot as plt
import pylab as pl
import math

def controlUser(budget, taskSet, userCost, userTaskSet, totalTaskNum, indexUserNum, userTaskNumDis, userSetDict,
                userSetSubsetDict):
    initUser = 20
    x_1 = np.array([])
    y_1 = np.array([])
    y_2 = np.array([])
    y_3 = np.array([])
    Y_1 = np.array([])
    Y_2 = np.array([])
    # 从50到800
    for i in range(indexUserNum):
        num = 40 + i * initUser
        print("user 数量为：:", num)
        userCostTemp_1 = userCost[:num]
        userTaskSetTemp_1 = userTaskSet

        userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM = SM.SingleMindedAlg(budget/2 , taskSet,
                                                                                      userCostTemp_1,
                                                                                      userTaskSetTemp_1,
                                                                                      totalTaskNum, num,
                                                                                      userTaskNumDis)
        print("SM总价值：", finalValue_SM, "SM平均收益：", averageUtility_SM)
        # userCostTemp_2 = userCost[:num]
        # userTaskSetTemp_2 = userTaskSet[:, :num]
        userPayment_MM, finalValue_MM, S_w_MM, averageUtility_MM = MM.MultiMindedAlg(budget, taskSet, userCost, num,
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
    indexUserNum = int(((totalUserNum - 40) / 20) + 1)
    SM_platformUtility_1, SM_platformUtility_2, MM_platformUtility_2, GM_platformUtility_2, \
    SM_averageUtility_2, MM_averageUtility_2 = np.zeros((indexUserNum,),
                                                        dtype=np.float), np.zeros(
        (indexUserNum,), dtype=np.float), np.zeros((indexUserNum,), dtype=np.float), np.zeros((indexUserNum,),
                                                                                              dtype=np.float), np.zeros(
        (indexUserNum,), dtype=np.float), np.zeros((indexUserNum,),
                                                   dtype=np.float)
    # 总共执行reNum组随机数据然后去平均值画图
    for i in range(reNum):
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
    # 画图-platformUtility
    plt.figure()
    # plt.plot(SM_platformUtility_1 / reNum, math.log(SM_platformUtility_2 / reNum,10), 'r', marker='x',
    #          label='SPBF-SM')  # use pylab to plot x and y : Give your plots names
    # plt.plot(SM_platformUtility_1 / reNum, math.log(MM_platformUtility_2/ reNum,10), 'g', marker='.', label='SPBF-MM')
    # plt.plot(SM_platformUtility_1 / reNum, math.log(GM_platformUtility_2 / reNum,10), 'b', marker='*', label='GM-SM')

    plt.plot(SM_platformUtility_1/reNum, SM_platformUtility_2/reNum , 'r', marker='x',
             label='SPBF-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1/reNum , MM_platformUtility_2 /reNum, 'g', marker='.', label='SPBF-MM')
    plt.plot(SM_platformUtility_1/reNum , GM_platformUtility_2/reNum , 'b', marker='*', label='GM-SM')

    plt.title('Impact of users')  # give plot a title
    plt.xlabel('Number of users')  # make axis labels
    plt.ylabel('Platform Utility')

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("platformUtility_users.pdf")
    plt.show()  # show the plot on the screen

    # 画图-averageUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_averageUtility_2 / reNum, 'r', marker='x',
             label='SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_averageUtility_2 / reNum, 'g', marker='.', label='MM')

    plt.title('Impact of users')  # give plot a title
    plt.xlabel('Number of users')  # make axis labels
    plt.ylabel('Average Utility')

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("avgUtility_users.pdf")
    plt.show()  # show the plot on the screen


def controlTask(budget, indexUserNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    initTask = 20
    x_1 = np.array([])
    y_1 = np.array([])
    y_2 = np.array([])
    y_3 = np.array([])
    Y_1 = np.array([])
    Y_2 = np.array([])
    # 从20到60
    for i in range(indexUserNum):
        totalTaskNum = 20 + i * initTask
        print("task 数量为：:", totalTaskNum)

        Data = dp.DataGenerate(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
        # taskSet = TaskSet(totalTaskNum, taskValueDis)
        taskSet = Data.TaskSet()
        # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
        userTaskSet, userCost = Data.UserTaskSet()
        # u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)

        userSetDict = Data.userSetDictCompute(userTaskSet)
        userSetSubsetDict = Data.userSetSubsetDictCompute(userSetDict)

        userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM = SM.SingleMindedAlg(budget / 2, taskSet,
                                                                                      userCost,
                                                                                      userTaskSet,
                                                                                      totalTaskNum, totalUserNum,
                                                                                      userTaskNumDis)
        print("SM总价值：", finalValue_SM, "SM平均收益：", averageUtility_SM)

        userPayment_MM, finalValue_MM, S_w_MM, averageUtility_MM = MM.MultiMindedAlg(budget, taskSet, userCost,
                                                                                     totalUserNum,
                                                                                     userSetDict, userSetSubsetDict)
        print("MM总价值：", finalValue_MM, "MM平均收益：", averageUtility_MM)

        userPayment_GM, finalValue_GM, S_w_GM, averageUtility_GM = GM.GreedyAlgSM(budget, taskSet, userCost,
                                                                                  userTaskSet, totalTaskNum,
                                                                                  totalUserNum)
        print("GM总价值：", finalValue_GM, "GM平均收益：", averageUtility_GM, "\n")

        x_1 = np.append(x_1, np.array([totalTaskNum]))
        y_1 = np.append(y_1, np.array([finalValue_SM]))
        y_2 = np.append(y_2, np.array([finalValue_MM]))
        y_3 = np.append(y_3, np.array([finalValue_GM]))
        Y_1 = np.append(Y_1, np.array([averageUtility_SM]))
        Y_2 = np.append(Y_2, np.array([averageUtility_MM]))
    return x_1, y_1, y_2, y_3, Y_1, Y_2


def doControlTask(reNum, budget, maxTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    # 初始设置参数
    # task 考虑的组数(user 考虑40-100，每次增加10)
    indexTaskNum = int(((maxTaskNum - 40) / 20) + 1)
    SM_platformUtility_1, SM_platformUtility_2, MM_platformUtility_2, GM_platformUtility_2, \
    SM_averageUtility_2, MM_averageUtility_2 = np.zeros((indexTaskNum,),
                                                        dtype=np.float), np.zeros(
        (indexTaskNum,), dtype=np.float), np.zeros((indexTaskNum,), dtype=np.float), np.zeros((indexTaskNum,),
                                                                                              dtype=np.float), np.zeros(
        (indexTaskNum,), dtype=np.float), np.zeros((indexTaskNum,),
                                                   dtype=np.float)
    # 执行并画图
    for i in range(reNum):
        # 控制user数量的图
        SM_pu_1, SM_pu_2, MM_pu_2, GM_pu_2, SM_au_2, MM_au_2 = controlTask(budget, indexTaskNum, taskValueDis,
                                                                           totalUserNum, userCosPerValueDis,
                                                                           userTaskNumDis)

        # print("GM_pu_1", GM_pu_1, "\n")
        # print("GM_platformUtility_1", GM_platformUtility_1, "\n")

        SM_platformUtility_1 = SM_platformUtility_1 + SM_pu_1
        SM_platformUtility_2 = SM_platformUtility_2 + SM_pu_2
        MM_platformUtility_2 = MM_platformUtility_2 + MM_pu_2
        GM_platformUtility_2 = GM_platformUtility_2 + GM_pu_2
        SM_averageUtility_2 = SM_averageUtility_2 + SM_au_2
        MM_averageUtility_2 = MM_averageUtility_2 + MM_au_2

        # print(SM_platformUtility_1,SM_platformUtility_2,MM_platformUtility_1,MM_platformUtility_2,SM_averageUtility_1,SM_averageUtility_2,MM_averageUtility_1,MM_averageUtility_2)

    # 画图-platformUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_platformUtility_2 / reNum, 'r', marker='x',
             label='SPBF-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_platformUtility_2 / reNum, 'g', marker='.', label='SPBF-MM')
    plt.plot(SM_platformUtility_1 / reNum, GM_platformUtility_2 / reNum, 'b', marker='*', label='GM-SM')

    plt.title('Impact of Tasks')  # give plot a title
    plt.xlabel('Number of Tasks')  # make axis labels
    plt.ylabel('Platform Utility')

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("platformUtility_tasks.pdf")
    plt.show()  # show the plot on the screen

    # 画图-averageUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_averageUtility_2 / reNum, 'r', marker='x',
             label='SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_averageUtility_2 / reNum, 'g', marker='.', label='MM')

    plt.title('Impact of Tasks')  # give plot a title
    plt.xlabel('Number of Tasks')  # make axis labels
    plt.ylabel('Average Utility')

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("avgUtility_tasks.pdf")
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
        budget = 40 + i * initUser
        print("budget为:", budget)

        userPayment_SM, finalValue_SM, S_w_SM, averageUtility_SM = SM.SingleMindedAlg(budget / 2, taskSet,
                                                                                      userCost,
                                                                                      userTaskSet,
                                                                                      totalTaskNum, totalUserNum,
                                                                                      userTaskNumDis)
        print("SM总价值：", finalValue_SM, "SM平均收益：", averageUtility_SM)
        # userCostTemp_2 = userCost[:num]
        # userTaskSetTemp_2 = userTaskSet[:, :num]
        userPayment_MM, finalValue_MM, S_w_MM, averageUtility_MM = MM.MultiMindedAlg(budget, taskSet, userCost,
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
    indexBudget = int(((maxBudget - 40) / 40) + 1)
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

    # 画图-platformUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_platformUtility_2 / reNum, 'r', marker='x',
             label='SPBF-SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_platformUtility_2 / reNum, 'g', marker='.', label='SPBF-MM')
    plt.plot(SM_platformUtility_1 / reNum, GM_platformUtility_2 / reNum, 'b', marker='*', label='GM-SM')

    plt.title('Impact of budget')  # give plot a title
    plt.xlabel('Budget')  # make axis labels
    plt.ylabel('Platform Utility')

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("platformUtility_budget.pdf")
    plt.show()  # show the plot on the screen

    # 画图-averageUtility
    plt.figure()
    plt.plot(SM_platformUtility_1 / reNum, SM_averageUtility_2 / reNum, 'r', marker='x',
             label='SM')  # use pylab to plot x and y : Give your plots names
    plt.plot(SM_platformUtility_1 / reNum, MM_averageUtility_2 / reNum, 'g', marker='.', label='MM')

    plt.title('Impact of budget')  # give plot a title
    plt.xlabel('Budget')  # make axis labels
    plt.ylabel('Average Utility')

    # pl.xlim(10.0, 35.0)  # set axis limits
    # pl.ylim(35.0, 50.0)
    plt.legend()
    plt.savefig("avgUtility_budget.pdf")
    plt.show()  # show the plot on the screen


if __name__ == '__main__':
    # 初始设置参数
    reNum = 20
    budget = 200
    totalTaskNum = 150
    taskValueDis = 5
    totalUserNum =300
    userCosPerValueDis = 10
    userTaskNumDis = 5

    doControlUser(reNum, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)

    # # 初始设置参数
    # reNum = 50
    # budget = 400
    # totalTaskNum = 300
    # taskValueDis = 5
    # totalUserNum = 200
    # userCosPerValueDis = 10
    # userTaskNumDis = 5
    # maxTaskNum = totalTaskNum
    # doControlTask(reNum, budget, maxTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)

    # reNum = 50
    # budget = 320
    # totalTaskNum = 150
    # taskValueDis = 5
    # totalUserNum = 200
    # userCosPerValueDis = 10
    # userTaskNumDis = 5
    #
    # doControlBudget(reNum, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis)
