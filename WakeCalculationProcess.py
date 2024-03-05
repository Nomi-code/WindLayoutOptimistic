import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import root
from math import pi

warnings.filterwarnings('error')
originalWindData = pd.read_excel('data/风数据.xlsx')


class KcCcomputer:
    __keys = originalWindData.keys()
    ks = {}
    cs = {}

    def computeKC(self):
        for key in self.__keys:
            u0i = originalWindData[key]
            u0i = u0i[~np.isnan(u0i)]

            def kGreatLikelihood(k):
                lnu0 = np.log(u0i)
                u0k = np.power(u0i, k)

                firstParamDenominator = np.dot(u0k, lnu0)
                firstParamNumerator = np.sum(u0k)

                secondParamDenominator = np.sum(lnu0)
                secondParamNumerator = len(u0i)

                return 1 / k - firstParamDenominator / firstParamNumerator + secondParamDenominator / secondParamNumerator

            def cCalculationProcess(u0, k):
                denominator = np.sum(np.log(u0))
                numerator = len(u0)

                return np.power(denominator / numerator, 1 / k)

            self.ks[key] = root(kGreatLikelihood, [1]).x[0]
            self.cs[key] = cCalculationProcess(u0i, self.ks[key])


class Env:
    __m = 8
    __n = 10
    __h = 70
    __D = 80
    __inclination = 7.2
    __distanceBwtFan = 7 * __D
    thetaDirectionMapping = {
        'N': 90,
        'NNE': 120,
        'NEE': 150,
        'E': 180,
        'EES': 210,
        'ESS': 240,
        'S': 270,
        'SSW': 300,
        'SWW': 330,
        'W': 0,
        'WWN': 30,
        'WNN': 60,
    }
    fansNumber = __m * __n
    positionData = None

    def __initPositionData(self):
        self.positionData = []

        offsetX = self.__distanceBwtFan * np.sin(self.__inclination)
        offsetY = self.__distanceBwtFan * np.cos(self.__inclination)
        for i in range(self.__m):
            for j in range(self.__n):
                x = j * self.__distanceBwtFan + offsetX * i
                y = 0 + offsetY * i
                self.positionData.append([x, y, self.__h])

        self.positionData = np.asarray(self.positionData)

    def __initTransform(self, theta):
        transformer = np.asarray([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        xyOnly = self.positionData[:, :2].copy()
        xyTransform = xyOnly.dot(transformer)
        self.positionData[:, :2] = xyTransform

    def __init__(self, theta):
        self.__initPositionData()
        self.__initTransform(theta)


class Fan:
    rRi = 40
    ARi = np.pi * np.power(rRi, 2)
    ThrustCofMapping = {
        4: 0.818,
        5: 0.806,
        6: 0.804,
        7: 0.805,
        8: 0.806,
        9: 0.807,
        10: 0.793,
        11: 0.739,
        12: 0.709,
        13: 0.409,
        14: 0.314,
        15: 0.249,
        16: 0.202,
        17: 0.167,
        18: 0.140,
        19: 0.119,
        20: 0.102,
        21: 0.088,
        22: 0.077,
        23: 0.067,
        24: 0.060,
        25: 0.053
    }

    def __getThrustMapping(self):
        mapping = np.vectorize(
            lambda windSpeed: 0 if windSpeed < 4 or windSpeed > 25 else self.__ThrustCofMapping[windSpeed])
        return mapping

    def getThrust(self, u0):
        mapping = self.__getThrustMapping()
        return mapping(u0)


def RWiCalculationProcess(rRi, CTi):
    para = np.sqrt(1 - CTi)

    return rRi * np.sqrt(
        (1 - para) / (2 * para)
    )


def alphaiCalculationProcess(hi, z0):
    return 0.5 / np.log(hi / z0)


def DijAndDRWijCalculationProcess(positionData, theta):
    """ input shape is [nT * 3] (x, y, h)
        return distances (Dij) and distancesRW (DRWij) [nT * nT]
    """
    positionData = np.asarray(positionData)
    nT = positionData.shape[0]

    distances = np.zeros((nT, nT))
    distancesRW = np.zeros((nT, nT))

    for i in range(nT):
        for j in range(nT):
            xi, yi, hi = positionData[i]
            xj, yj, hj = positionData[j]

            xDiff = xi - xj
            yDiff = yi - yj
            hDiff = hi - hj

            xSquare = np.power(xDiff, 2)
            ySquare = np.power(yDiff, 2)
            hSquare = np.power(hDiff, 2)

            Dij = np.abs(xDiff * np.cos(theta) + yDiff * np.sin(theta))
            DRWij = np.sqrt(
                xSquare + ySquare + hSquare - np.power(Dij, 2)
            )

            distances[i][j] = Dij
            distancesRW[i][j] = DRWij
    return [distances, distancesRW]


def RWijCalculationProcess(Dij, RWi, positionData, z0):
    h = positionData[:, 2]
    alpha = 0.5 / np.log(h / z0)
    nT = Dij.shape[0]
    RWij = np.zeros(Dij.shape)
    """
    TODO:
    RWij会有影响
    """
    for i in range(nT):
        for j in range(i + 1, nT):
            RWij[i][j] = alpha[i] * Dij[i][j] + RWi
    # RWij = np.multiply(alpha, Dij) + RWi
    return RWij


def thetaWijAndThetaRijCalculationProcess(DRWij, RWij, rRi):
    paraDenominator1 = np.power(DRWij, 2)
    paraDenominator2 = np.power(RWij, 2)
    paraDenominator3 = np.power(rRi, 2)

    paraNumerator = 2 * np.multiply(RWij, DRWij)
    paraNumerator[paraNumerator == 0] = np.inf
    paraNumerator = np.power(paraNumerator, -1)
    paraNumerator[np.isinf(paraNumerator)] = 0

    paraThetaWij = np.zeros(paraDenominator1.shape)
    paraThetaRij = np.zeros(paraDenominator1.shape)

    c1 = (RWij + rRi <= DRWij)
    c2 = (RWij - rRi >= DRWij)
    c3 = ~(c1 | c2)

    paraThetaWij[c1] = 0
    paraThetaWij[c2] = 0
    paraThetaRij[c1] = 0
    paraThetaRij[c2] = 0

    paraThetaWij[c3] = np.multiply((paraDenominator1[c3] + paraDenominator2[c3] - paraDenominator3), paraNumerator[c3])
    paraThetaRij[c3] = np.multiply((paraDenominator1[c3] - paraDenominator2[c3] + paraDenominator3), paraNumerator[c3])

    thetaWij = 2 * np.arcsin(paraThetaWij)
    thetaRij = 2 * np.arccos(paraThetaRij)
    return [thetaWij, thetaRij]


def ARWijCalculationProcess(RWij, rRi, DRWij, thetaWij, thetaRij):
    ARWij = np.asarray(RWij).copy()

    conditionIndex1 = (RWij + rRi <= DRWij)
    conditionIndex2 = (RWij - rRi >= DRWij)
    conditionIndex3 = ~(conditionIndex1 | conditionIndex2)

    ARWij[conditionIndex1] = 0
    ARWij[conditionIndex2] = pi * np.power(rRi, 2)
    ARWij[conditionIndex3] = 0.5 * (
            np.multiply(
                np.power(RWij[conditionIndex3], 2),
                thetaWij[conditionIndex3] - np.sin(thetaWij[conditionIndex3])
            ) + \
            np.power(rRi, 2) * (thetaRij[conditionIndex3] - np.sin(thetaRij[conditionIndex3]))
    )
    ARWij = np.triu(ARWij, 1)
    return ARWij


def Ui0CalculationProcess(u0, hi, h0, z0):
    ui0 = u0 * (
            np.log(hi / z0) / np.log(h0 / z0)
    )
    return ui0


def deltaUijCalculationProcess(CsubT, RWi, RWij, ARWij, ARi):
    RWij[RWij == 0] = np.inf

    para1 = 1 - np.sqrt(1 - CsubT)
    para2 = np.power(np.multiply(RWi, np.power(RWij, -1)), 2)
    para3 = np.multiply(ARWij, np.power(ARi, -1))
    deltaUij = para1 * np.multiply(para2, para3)
    deltaUij[np.isnan(deltaUij)] = 0
    return deltaUij


def deltaUijCalculation(positionData, rRi, CTi, theta, CsubT, ARi, z0):
    RWi = RWiCalculationProcess(rRi, CTi)
    Dij, DRWij = DijAndDRWijCalculationProcess(positionData, theta)
    RWij = RWijCalculationProcess(Dij, RWi, positionData, z0)
    thetaWij, thetaRij = thetaWijAndThetaRijCalculationProcess(DRWij, RWij, rRi)
    ARWij = ARWijCalculationProcess(RWij, rRi, DRWij, thetaWij, thetaRij)
    deltaUij = deltaUijCalculationProcess(CsubT, RWi, RWij, ARWij, ARi)
    return deltaUij


def deltaUiCalculation(deltaUij):
    deltaUij = np.power(deltaUij, 2)
    deltaUi = np.sum(deltaUij, axis=0)
    return np.asarray(deltaUi)


def UiCalculationProcess(positionData, rRi, CTi, theta, CsubT, ARi, u0, h0, z0):
    deltaUij = deltaUijCalculation(positionData, rRi, CTi, theta, CsubT, ARi, z0)
    deltaUi = deltaUiCalculation(deltaUij)
    ui0 = Ui0CalculationProcess(u0, positionData[:, 2], h0, z0)
    ui0 = ui0 - deltaUi
    return ui0


h0, z0 = 60, 0.0002
fan = Fan()
#
# plt.scatter(env.positionData[:, 0], env.positionData[:, 1])
# plt.show()

for direction in originalWindData.keys():
    # if direction != 'W':
    #     continue
    us = np.asarray(originalWindData[direction])
    us = us[~np.isnan(us)]
    uis = []
    for u0 in us:
        theta = Env.thetaDirectionMapping[direction]
        env = Env(theta)
        CT = 0 if u0 < 4 or u0 > 25 else fan.ThrustCofMapping[int(u0)]
        try:
            ui = UiCalculationProcess(env.positionData, fan.rRi, CT, theta, CT, fan.ARi, u0, h0, z0)
        except Warning as w:
            print(w.args)
            breakpoint = 1
