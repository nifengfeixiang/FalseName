import numpy as np
from random import sample
from scipy.stats import uniform



def InitialSetting(budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
    """
    initial all the settings
    :param budget: 预算
    :param totalTaskNum: 所有可能总共task的数量
    :param taskValueDis: 每个task的价值最大值
    :param totalUserNum: 用户的数量
    :param userCosPerValueDis: 用户每个任务的单价最大值
    :param userTaskNumDis:  每个用户task数量的最大值
    :return:
    """
    return budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis



class DataGenerate:
    def __init__(self, budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis):
        self.budget=budget
        self.totalTaskNum=totalTaskNum
        self.taskValueDis=taskValueDis
        self.totalUserNum=totalUserNum
        self.userCostPerValueDis=userCosPerValueDis
        self.userTaskNumDis=userTaskNumDis

    def UniformDis(self, size, dis):
        """
        :param dis: distribution range
        :param size: the number of variables
        :return: random number
        """
        set = np.array(uniform.rvs(1, dis, size)).astype(int)
        return set

    def TaskSet(self):
        """
        Generate the task set;
        :param SetSize: the size of task;
        :param valueDistribution: the value distribution of each task;(uniform distribution)
        """
        # taskSet=np.zeros(shape=(1,Distribution),dtype=float)
        taskSet = self.UniformDis(self.totalTaskNum, self.taskValueDis)
        return taskSet

    def UserTaskSet(self):
        """
        generate the user set
        :param SetSize: user set size
        :param costPerTaskDis: cost per task for each user distribution
        :param maxNum: max number of task set of each user
        :param taskset: task set containing all tasks
        :return:
        """

        # gengrate the task query
        # taskNum = self.totalTaskNum
        task = np.zeros((self.totalTaskNum,), dtype=np.int)
        for i in range(self.totalTaskNum):
            task[i] = i
        task = task.tolist()

        # generate the task set of each user
        userTaskSet = np.zeros(shape=(self.totalTaskNum, self.totalUserNum), dtype=np.int)
        eachUserTaskSetSize = np.array(uniform.rvs(1, self.userTaskNumDis , self.totalUserNum)).astype(int)
        # print("随机生成的user任务size：",eachUserTaskSetSize)
        for i in range(self.totalUserNum):
            userTask = sample(task, eachUserTaskSetSize[i])
            # size = len(userTask)
            for item in userTask:
                userTaskSet[item][i] = 1

        # genetate the cost
        userCost = np.zeros((self.totalUserNum,), dtype=np.float)
        for i in range(self.totalUserNum):
            userCost[i] = round(uniform.rvs(1, self.userCostPerValueDis, 1)[0], 2)

        return userTaskSet, userCost

    # 得到每个user的任务集合的字典
    def userSetDictCompute(self,userTaskSet):
        userSetDict = {}
        for i in range(self.totalUserNum):
            tempSet = set()
            for j in range(self.totalTaskNum):
                if (userTaskSet[j][i] == 1):
                    tempSet.add(j)
            userSetDict[i] = tempSet
        return userSetDict

    # 计算user集合的除去空集的所有子集的字典表示，用list表示所有的子集
    def userSetSubsetDictCompute(self,userSetDict):
        userSetSubsetDict = {}
        for user in range(self.totalUserNum):
            items = list(userSetDict[user])
            # generate all combination of N items
            N = len(items)
            # enumerate the 2**N possible combinations
            set_all = []
            for i in range(2 ** N):
                combo = []
                for j in range(N):
                    if (i >> j) % 2 == 1:
                        combo.append(items[j])
                set_all.append(combo)
            userSetSubsetDict[user] = set_all
        return userSetSubsetDict

# def UniformDis(size, dis):
#     """
#     :param dis: distribution range
#     :param size: the number of variables
#     :return: random number
#     """
#     set = np.array(uniform.rvs(1, dis - 1, size)).astype(int)
#     return set
#
#
# def TaskSet(SetSize, valueDistribution):
#     """
#     Generate the task set;
#     :param SetSize: the size of task;
#     :param valueDistribution: the value distribution of each task;(uniform distribution)
#     """
#     # taskSet=np.zeros(shape=(1,Distribution),dtype=float)
#     taskSet = UniformDis(SetSize, valueDistribution)
#     return taskSet
#
#
# def UserSet(UserSize, costPerTaskDis, maxNum, taskSet):
#     """
#     generate the user set
#     :param SetSize: user set size
#     :param costPerTaskDis: cost per task for each user distribution
#     :param maxNum: max number of task set of each user
#     :param taskset: task set containing all tasks
#     :return:
#     """
#
#     # gengrate the task query
#     taskNum = np.size(taskSet)
#     task = np.zeros((taskNum,), dtype=np.int)
#     for i in range(taskNum):
#         task[i] = i
#     task = task.tolist()
#
#     # generate the task set of each user
#     userTaskSet = np.zeros(shape=(np.size(taskSet), UserSize), dtype=np.int)
#     eachUserTaskSetSize = np.array(uniform.rvs(1, maxNum - 1, UserSize)).astype(int)
#     for i in range(UserSize):
#         userTask = sample(task, eachUserTaskSetSize[i])
#         size = len(userTask)
#         for j in range(size):
#             userTaskSet[userTask[j]][i] = 1
#
#     # genetate the cost
#     userCost = np.zeros((UserSize,), dtype=np.float)
#     for i in range(UserSize):
#         userCost[i] = round(uniform.rvs(0, costPerTaskDis, 1)[0], 2)
#
#     return userTaskSet, userCost


def Argmin(u, u_w, R, userCost, userTaskSet):
    """
    计算Argmin
    :param u:
    :param u_w:
    :param R:
    :param userCost:
    :param userTaskSet:
    :return:
    """
    cost = userCost.copy()
    taskSize = np.shape(userTaskSet)[0]

    # 根据能够u\u_w重新处理cost序列
    for i in u_w:
        cost[i] = 10
    for i in (u - u_w):
        userSet = set()
        for j in range(taskSize):
            if userTaskSet[j][i] == 1:
                userSet.add(j)
        if len(userSet - R) == 0:
            cost[i] = 10
    minCost = np.min(cost)
    minCostIndex = np.where(cost == minCost)[0][0]
    setNum = np.sum(userTaskSet[:,minCostIndex])
    userSet = set()
    for i in range(taskSize):
        if userTaskSet[i][minCostIndex] == 1:
            userSet.add(i)
    return minCost, minCostIndex, setNum, userSet


def WinnerSelection(u, u_w, R, userCost, userTaskSet, budget):
    """
    winner select
    :param u:
    :param u_w:
    :param R:
    :param userCost:
    :param userTaskSet:
    :param budget:
    :return: u_w,winner set;R,winning task set
    """
    # taskSize = np.size(userTaskSet)[0]  # 所有task的数量
    # compute \argmin
    minCost, minCostIndex, setNum, userSet = Argmin(u, u_w, R, userCost, userTaskSet)
    Num = setNum
    print(setNum)
    # winner selection
    while (minCost <= round((budget / Num), 2)):
        u_w.add(minCostIndex)
        R = R | userSet
        Num = Num + setNum
        minCost, minCostIndex, setNum, userSet = Argmin(u, u_w, R, userCost, userTaskSet)
    return u_w, R


def PaymentScheme(user, u, u_w, R, userCost, userTaskSet, budget):
    minCost, minCostIndex, setNum, userSet = Argmin(u, u_w, R, userCost, userTaskSet)
    Num = setNum
    p = 0
    # winner selection
    while (minCost <= budget / Num):
        u_w.add(minCostIndex)
        R = R | userSet
        Num = Num + setNum
        minCost, minCostIndex, setNum, userSet = Argmin(u, u_w, R, userCost, userTaskSet)
    for i in range(np.shape(userTaskSet)[0]):
        if userTaskSet[i][user] == 1:
            for j in range(np.shape(userTaskSet)[1]):
                tempSet=set()
                tempSet.add(j)
                if (userTaskSet[i][j] == 1) and (j != user) and (tempSet in u_w):
                    p = max(p, userCost[j])
    p = p * np.sum(userTaskSet[:,user])
    return p


def SM(budget, taskSet, userTaskSet, userCost):
    """
    Multi-minded algorithm;
    :param budget: budget
    :param taskSet: taskset
    :param userTaskSet: usertaskset
    :param userCost: percost
    :return: obtained total value
    """
    # 初始化集合u,u_w,R
    userNum = np.shape(userTaskSet)[1]
    u = set()
    for i in range(userNum):
        u.add(i)
    R = set()
    u_w = set()

    taskSize = np.size(taskSet)  # 所有task的数量
    # winner select
    u_w, R = WinnerSelection(u, u_w, R, userCost, userTaskSet, budget)
    # compute total value
    totalValue = 0
    for task in R:
        totalValue = totalValue + taskSet[task]

    # payment scheme
    P = np.zeros((np.shape(userTaskSet)[1],), dtype=np.float)
    u_copy = u.copy()
    for user in u_w:
        u_user = u_copy.copy()
        u_user.remove(user)
        u_w_prime = set()
        R_prime = set()
        P[user] = PaymentScheme(user, u_user, u_w_prime, R_prime, userCost, userTaskSet, budget)
    return u_w, R, P, totalValue





if __name__ == '__main__':
    # budget = 20
    # totalTaskNum = 10
    # taskValueDis = 20
    # totalUserNum = 10
    # userCosPerValueDis = 2.5
    # userTaskNumDis = 4
    budget, totalTaskNum, taskValueDis, totalUserNum, userCosPerValueDis, userTaskNumDis = InitialSetting(20, 20, 30,
                                                                                                          10, 2.5, 4)

    Data=DataGenerate(20, 20, 30, 10, 2.5, 4)
    # taskSet = TaskSet(totalTaskNum, taskValueDis)
    taskSet=Data.TaskSet()
    # userTaskSet, userCost = UserSet(totalUserNum, userCosPerValueDis, userTaskNumDis, taskSet)
    userTaskSet, userCost=Data.UserTaskSet()
    u_w, R, p, totalValue = SM(budget, taskSet, userTaskSet, userCost)
    print("taskSet:", taskSet, "\n")
    print("userTaskSet:", userTaskSet, "\n")
    print("userCost:", userCost, "\n")
    print("Winner:", u_w, "\n")
    print("Total value:", totalValue)
