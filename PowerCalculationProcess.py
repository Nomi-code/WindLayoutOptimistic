import pandas as pd
import numpy as np
from math import pi


class PowerModel:
    __uin = 4
    __uout = 25
    __ur = 15
    __Pr = 0

    @staticmethod
    def __CpCalculationProcess(ui, cd1, cd2, cd3):
        Cp = np.asarray(ui).copy()
        Cp[cd1] = 0
        Cp[cd2] = 0.5 * np.sin(pi / 22 * (Cp[cd2] - 4))
        Cp[cd3] = 0.5 * np.cos(pi / 20 * (Cp[cd3] - 15))

        return Cp

    def __PiCalculationProcess(self, ui, rRi, ro):
        uin = self.__uin
        uout = self.__uout
        ur = self.__ur
        Pr = self.__Pr

        conditionIndex1 = ((ui < uin) | (ui > uout))
        conditionIndex2 = ((uin <= ui) & (ui < ur))
        conditionIndex3 = (~conditionIndex1 & ~conditionIndex2)

        Cp = self.__CpCalculationProcess(ui, conditionIndex1, conditionIndex2, conditionIndex3)
        Pi = np.asarray(ui).copy()

        Pi[conditionIndex1] = 0
        Pi[conditionIndex2] = 0.5 * pi * ro * Cp[conditionIndex2] \
                              * np.power(rRi[conditionIndex2], 2) \
                              * np.power(ui[conditionIndex2], 3)
        Pi[conditionIndex3] = Pr

        return Pi

    def AEPCalculationProcess(self, windData, rRi, ro):
        AEP = 0

        for direction in windData.keys():
            ui = windData[direction]
            ui = ui[~np.isnan(ui)]
            AEP += np.sum(self.__PiCalculationProcess(ui, rRi, ro))

        assert AEP != 0
        return AEP


# a = PowerModel()

a = np.asarray([
    [1, 2],
    [1, 2]
])

print(np.sum(a, axis=1))