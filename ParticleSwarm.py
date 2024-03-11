import warnings
import numpy as np
import pandas as pd
import random as rd
import sys
from sko.PSO import PSO
from WakeCalculationProcess import PowerModel, WakeCalculationModel, Env, Fan

# warnings.filterwarnings('error')
originalWindData = pd.read_excel('data/风数据.xlsx')


class ParticleSwarm:
    __swarmNumber: int
    __dimensionNumber: int = 3
    __inertialWeight: float
    __inertialWeightMax: float
    __inertialWeightMin: float
    __iteration: int
    __stepLength: float
    __c1: float = 1.6
    __c2: float = 1.8
    __coordinate: np.array
    __velocity: np.array
    __r1: float
    __r2: float
    __pibest: np.array
    __pbgest: np.array
    __xmax: float
    __xmin: float
    __ymax: float
    __ymin: float
    __vmax: float
    __vmin: float
    __h1: float
    __h2: float
    __fvalue: float = sys.float_info.max
    __locationHis: np.array = np.asarray([])
    coordinateHistory = []

    def __subjectTo(self, xi, yi, hi):
        # TODO
        return

    def __updateVelocity(self):
        self.__velocity = self.__inertialWeight * self.__velocity + \
                          self.__c1 * self.__r1 * (self.__pibest - self.__coordinate) + \
                          self.__c2 * self.__r2 * (self.__pbgest - self.__coordinate)

    def __updateCoordinate(self):
        self.__coordinate = self.__coordinate + self.__velocity

    def __updateInertialWeight(self, currentIter):
        self.__inertialWeight = \
            self.__inertialWeightMax - \
            (currentIter / self.__iteration) * (self.__inertialWeightMax - self.__inertialWeightMin)

    def __init__(self, swarmNumber,
                 inertialWeight, inertialWeightMax, inertialWeightMin,
                 iteration, stepLength,
                 xmax, xmin, ymax, ymin, vmax, vmin, h1, h2):
        self.__swarmNumber = swarmNumber
        self.__inertialWeight = inertialWeight
        self.__inertialWeightMax = inertialWeightMax
        self.__inertialWeightMin = inertialWeightMin
        self.__iteration = iteration
        self.__stepLength = stepLength
        self.__xmax = xmax
        self.__xmin = xmin
        self.__ymax = ymax
        self.__ymin = ymin
        self.__vmax = vmax
        self.__vmin = vmin
        self.__h1 = h1
        self.__h2 = h2

        self.__r1 = rd.random()
        self.__r2 = rd.random()

        xDiff = self.__xmax - self.__xmin
        yDiff = self.__ymax - self.__ymin
        vDiff = self.__vmax - self.__vmin

        self.__coordinate = np.random.rand(self.__swarmNumber, self.__dimensionNumber)
        self.__coordinate[:, 0] = xDiff * self.__coordinate[:, 0] + self.__xmin
        self.__coordinate[:, 1] = yDiff * self.__coordinate[:, 1] + self.__ymin
        self.__coordinate[:, 2] = [self.__h1 if h0 > 0.5 else self.__h2 for h0 in self.__coordinate[:, 2]]
        self.__velocity = np.random.rand(self.__swarmNumber, self.__dimensionNumber) * vDiff + vmin

        self.__pbgest = np.zeros((self.__swarmNumber, self.__dimensionNumber))
        self.__pibest = self.__coordinate.copy()

    def startIterate(self):
        for currentIter in range(self.__iteration):
            UiMap = {}
            # 将Ui计算出来
            for direction in originalWindData.keys():
                us = np.asarray(originalWindData[direction])
                us = us[~np.isnan(us)]
                theta = Env.thetaDirectionMapping[direction]
                env = Env(theta, self.__coordinate)
                uis = []
                for u0 in us:
                    Ct = Fan.getThrust(u0)
                    ui = WakeCalculationModel.UiCalculationProcess(
                        env.positionData, Fan.rRi, Ct, theta, Ct, Fan.ARi, u0,
                        WakeCalculationModel.h0, WakeCalculationModel.z0
                    )
                    uis.append(ui)
                UiMap[direction] = uis
            AepMap = PowerModel.AEPmapCalculationProcess(UiMap, Fan.rRi, 1.205)
            CostMap = PowerModel.costMapCalculation(self.__coordinate)

            AEP = np.sum([AepMap[k] for k in AepMap.keys()])
            Cost = np.sum([CostMap[k] for k in CostMap.keys()])

            fResult = AEP / Cost
            if fResult < self.__fvalue:
                np.append(self.__locationHis, self.__coordinate)
                self.__fvalue = fResult
            self.__updateCoordinate()
            self.__updateVelocity()
            self.__updateInertialWeight(currentIter)
            self.coordinateHistory.append(self.__coordinate)


particleSwarm = ParticleSwarm(
    80, 0.1, 1, 0, 5, 0.2,
    500, 0, 500, 0, 0, 10, 60, 60
)

particleSwarm.startIterate()

print(particleSwarm.coordinateHistory)