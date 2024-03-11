import pandas as pd
import numpy as np
import warnings
import json
import csv
from scipy.optimize import root
from math import pi

# warnings.filterwarnings('error')
originalWindData = pd.read_excel('data/新高度70风数据.xlsx')


class Utils:
    @staticmethod
    def str2list(string: str):
        string = string.strip('[')
        string = string.strip(']')
        string = string.split(' ')
        return [float(s) for s in string]

    @staticmethod
    def processWindSpeed(windData):
        uimap = {}
        for direction in windData.keys():
            col = windData[direction]
            for i in range(len(col)):
                uimap[direction][i] = Utils.str2list(col[i])
        return uimap


class KcComputer:
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

        offsetX = self.__distanceBwtFan * np.sin(np.radians(self.__inclination))
        offsetY = self.__distanceBwtFan * np.cos(np.radians(self.__inclination))
        for i in range(self.__m):
            for j in range(self.__n):
                x = j * self.__distanceBwtFan + offsetX * i
                y = 0 + offsetY * i
                self.positionData.append([x, y, 70])

        self.positionData = np.asarray(self.positionData)

    def __initTransform(self, theta):
        transformer = np.asarray([
            [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
            [np.sin(np.radians(theta)), np.cos(np.radians(theta))]
        ])
        xyOnly = self.positionData[:, :2].copy()
        xyTransform = transformer.dot(xyOnly.T)
        xyTransform = xyTransform.T
        self.positionData[:, :2] = xyTransform

    def __init__(self, theta, positionData):
        if positionData is None:
            self.__initPositionData()
        else:
            self.positionData = positionData
        self.__initTransform(theta)


class Fan:
    rRi = 40
    ARi = np.pi * np.power(rRi, 2)
    __ThrustCofMapping = {
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

    @staticmethod
    def getThrust(u0):
        return 0 if u0 < 4 or u0 > 25 else Fan.__ThrustCofMapping[int(u0)]


class WakeCalculationModel:
    h0, z0 = 60, 0.0002

    @staticmethod
    def __RWiCalculationProcess(rRi, CTi):
        para = np.sqrt(1 - CTi)

        return rRi * np.sqrt(
            (1 + para) / (2 * para)
        )

    @staticmethod
    def __DijAndDRWijCalculationProcess(positionData, theta):
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

                Dij = np.abs(xDiff * np.cos(np.radians(0)) + yDiff * np.sin(np.radians(0)))

                DRWpara = xSquare + ySquare + hSquare - np.power(Dij, 2)
                DRWij = (0 if DRWpara <= 0 else np.sqrt(DRWpara))
                try:
                    DRWij = np.sqrt(DRWpara)
                except Warning as w:
                    print(w.args, theta, (xi, yi), (xj, yj), Dij, DRWpara)

                distances[i][j] = Dij
                distancesRW[i][j] = DRWij
        return [distances, distancesRW]

    @staticmethod
    def __RWijCalculationProcess(Dij, RWi, positionData, z0):
        h = positionData[:, 2]
        alpha = 0.5 / np.log(h / z0)
        nT = Dij.shape[0]
        RWij = np.zeros(Dij.shape)
        Dij[Dij < 1e-6] = 0
        for i in range(nT):
            for j in range(i + 1, nT):
                para = Dij[i][j]
                if para >= 12 * 80:
                    RWij[i][j] = -40
                else:
                    RWij[i][j] = alpha[i] * para + RWi
        return RWij

    @staticmethod
    def __thetaWijAndThetaRijCalculationProcess(DRWij, RWij, rRi):
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

        paraThetaWij[c3] = np.multiply((paraDenominator1[c3] + paraDenominator2[c3] - paraDenominator3),
                                       paraNumerator[c3])
        paraThetaRij[c3] = np.multiply((paraDenominator1[c3] - paraDenominator2[c3] + paraDenominator3),
                                       paraNumerator[c3])

        thetaWij = 2 * np.arcsin(np.radians(paraThetaWij))
        thetaRij = 2 * np.arccos(np.radians(paraThetaRij))
        return [thetaWij, thetaRij]

    @staticmethod
    def __ARWijCalculationProcess(RWij, rRi, DRWij, thetaWij, thetaRij):
        ARWij = np.asarray(RWij).copy()

        conditionIndex1 = (RWij + rRi <= DRWij)
        conditionIndex2 = (RWij - rRi >= DRWij)
        conditionIndex3 = ~(conditionIndex1 | conditionIndex2)

        ARWij[conditionIndex1] = 0
        ARWij[conditionIndex2] = pi * np.power(rRi, 2)
        ARWij[conditionIndex3] = 0.5 * (
                np.multiply(
                    np.power(RWij[conditionIndex3], 2),
                    thetaWij[conditionIndex3] - np.sin(np.radians(thetaWij[conditionIndex3]))
                ) + \
                np.power(rRi, 2) * (thetaRij[conditionIndex3] - np.sin(np.radians(thetaRij[conditionIndex3])))
        )
        ARWij = np.triu(ARWij, 1)
        return ARWij

    @staticmethod
    def __ui0CalculationProcess(u0, hi, h0, z0):
        ui0 = u0 * np.log(hi / z0) / np.log(h0 / z0)
        return ui0

    @staticmethod
    def __deltaUijCalculationProcess(CsubT, RWi, RWij, ARWij, ARi):
        RWij[RWij == 0] = np.inf

        para1 = 1 - np.sqrt(1 - CsubT)
        para2 = np.power(np.multiply(RWi, np.power(RWij, -1)), 2)
        para3 = np.multiply(ARWij, np.power(ARi, -1))
        deltaUij = para1 * np.multiply(para2, para3)
        deltaUij[np.isnan(deltaUij)] = 0
        return deltaUij

    @staticmethod
    def __deltaUijCalculation(positionData, rRi, CTi, theta, CsubT, ARi, z0):
        RWi = WakeCalculationModel.__RWiCalculationProcess(rRi, CTi)
        Dij, DRWij = WakeCalculationModel.__DijAndDRWijCalculationProcess(positionData, theta)
        RWij = WakeCalculationModel.__RWijCalculationProcess(Dij, RWi, positionData, z0)
        thetaWij, thetaRij = WakeCalculationModel.__thetaWijAndThetaRijCalculationProcess(DRWij, RWij, rRi)
        ARWij = WakeCalculationModel.__ARWijCalculationProcess(RWij, rRi, DRWij, thetaWij, thetaRij)
        deltaUij = WakeCalculationModel.__deltaUijCalculationProcess(CsubT, RWi, RWij, ARWij, ARi)
        return deltaUij

    @staticmethod
    def __deltaUiCalculation(deltaUij):
        deltaUij = np.power(deltaUij, 2)
        deltaUi = np.sum(deltaUij, axis=0)
        return np.asarray(deltaUi)

    @staticmethod
    def UiCalculationProcess(positionData, rRi, CTi, theta, CsubT, ARi, u0, h0, z0):
        deltaUij = WakeCalculationModel.__deltaUijCalculation(positionData, rRi, CTi, theta, CsubT, ARi, z0)
        deltaUi = WakeCalculationModel.__deltaUiCalculation(deltaUij)
        ui0 = WakeCalculationModel.__ui0CalculationProcess(u0, positionData[:, 2], h0, z0)
        ui0 = ui0 - deltaUi
        return ui0

    @staticmethod
    def computeUimap():
        fan = Fan()
        uimap = {}

        for direction in originalWindData.keys():
            us = np.asarray(originalWindData[direction])
            us = us[~np.isnan(us)]
            uis = []
            for u0 in us:
                theta = Env.thetaDirectionMapping[direction]
                env = Env(theta, None)
                CT = fan.getThrust(u0)
                ui = WakeCalculationModel.UiCalculationProcess(
                    env.positionData, fan.rRi, CT, theta, CT, fan.ARi, u0,
                    WakeCalculationModel.h0, WakeCalculationModel.z0
                )
                uis.append(ui)
            uimap[direction] = np.asarray(uis)
        return uimap


class PowerModel:
    __uin = 4
    __uout = 25
    __ur = 15
    __powerMapping = {
        4: 66.6,
        5: 154,
        6: 282,
        7: 460,
        8: 696,
        9: 996,
        10: 1341,
        11: 1661,
        12: 1866,
        13: 1958,
        14: 1988,
        15: 1997,
        16: 1999,
        17: 2000
    }
    __Ch = 1500
    __Cb = 593867

    @staticmethod
    def __getPowerByWindSpeed(ui):
        power = ui.copy()
        c1 = ui < 4
        c2 = ui > 17
        c3 = ~(c1 | c2)
        power[c1] = 0
        power[c2] = 2000
        power[c3] = [PowerModel.__powerMapping[int(u)] for u in power[c3]]
        return power

    @staticmethod
    def __CpCalculationProcess(ui, cd1, cd2, cd3):
        Cp = np.asarray(ui).copy()
        Cp[cd1] = 0
        Cp[cd2] = 0.5 * np.sin(pi / 22 * (Cp[cd2] - 4))
        Cp[cd3] = 0.5 * np.cos(pi / 20 * (Cp[cd3] - 15))

        return Cp

    @staticmethod
    def PiCalculationProcess(ui, rRi, ro):
        uin = PowerModel.__uin
        uout = PowerModel.__uout
        ur = PowerModel.__ur

        conditionIndex1 = ((ui < uin) | (ui > uout))
        conditionIndex2 = ((uin <= ui) & (ui < ur))
        conditionIndex3 = ~(conditionIndex1 | conditionIndex2)

        Cp = PowerModel.__CpCalculationProcess(ui, conditionIndex1, conditionIndex2, conditionIndex3)
        Pi = np.asarray(ui).copy()

        Pi[conditionIndex1] = 0
        Pi[conditionIndex2] = 0.5 * pi * ro * Cp[conditionIndex2] \
                              * np.power(rRi, 2) \
                              * np.power(ui[conditionIndex2], 3)
        Pi[conditionIndex3] = PowerModel.__getPowerByWindSpeed(ui[conditionIndex3])

        return Pi

    @staticmethod
    def AEPmapCalculationProcess(windData, rRi, ro):
        AepMap = {}

        for direction in windData.keys():
            ui = windData[direction]
            ui = np.asarray(ui)
            # vector calculation
            Aeps = PowerModel.PiCalculationProcess(ui, rRi, ro)
            AepMap[direction] = np.sum(Aeps)

        return AepMap

    @staticmethod
    def costMapCalculation(positionData):
        costMap = {}
        directions = Env.thetaDirectionMapping

        for direction in directions.keys():
            cost = 0
            theta = directions[direction]
            his = positionData[:, 2]
            for hi in his:
                cost += PowerModel.__Ch * hi + PowerModel.__Cb

            costMap[direction] = cost

        return costMap


def findOutWhyScaleGreaterThan1():
    Uimap = pd.read_excel('data/新风数据.xlsx')

    processedWindSpeed = Utils.str2list(Uimap['N'][0])
    processedWindSpeed = np.asarray(processedWindSpeed)
    originalWindSpeed = [originalWindData['N'][0] for i in range(80)]
    originalWindSpeed = np.asarray(originalWindSpeed)

    aep1 = PowerModel.PiCalculationProcess(processedWindSpeed, Fan.rRi, 1.205)
    aep2 = PowerModel.PiCalculationProcess(originalWindSpeed, Fan.rRi, 1.205)
    print(aep1, aep2)


def generateData():
    Uimap = WakeCalculationModel.computeUimap()
    for direction in Uimap.keys():
        mat = Uimap[direction]
        mat = np.asarray(mat)
        filepath = 'data/CalculateWindSpeedAfterWakeLosses/' + str(direction) + '.txt'
        np.savetxt(filepath, mat, delimiter=',')


def readData():
    Uimap = {}
    for direction in Env.thetaDirectionMapping.keys():
        filepath = 'data/CalculateWindSpeedAfterWakeLosses/' + str(direction) + '.txt'
        mat = np.loadtxt(filepath, delimiter=',')
        Uimap[direction] = mat
    return Uimap

# WakeCalculationModel.computeUimap()
Uimap = readData()
Aepmap = PowerModel.AEPmapCalculationProcess(Uimap, Fan.rRi, 1.205)
Aepmap_ = PowerModel.AEPmapCalculationProcess(originalWindData, Fan.rRi, 1.205)


scaleMap = {}

for direction in Aepmap.keys():
    aep = Aepmap[direction]
    aep_ = Aepmap_[direction] * 80

    scaleMap[direction] = (aep / aep_)
    temper = {
        'aep': aep,
        'aep_': aep_
    }
    print(direction, temper)

print(scaleMap)
breakpoint = 0
