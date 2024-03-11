import pandas as pd
import numpy as np
import warnings
import json
import csv
from scipy.optimize import root
from math import pi

originalWindData = pd.read_excel('data/风数据.xlsx')

# 风数据.xlsx 是h=60情况下的风速
def computeWineDataAtNewAltitudeAndSave(windData, newh):
    h0 = 60
    z0 = 0.0002
    for direction in windData.keys():
        ui = windData[direction]
        windData[direction] = ui * np.log(newh / z0) / np.log(h0 / z0)

    pd.DataFrame(windData).to_excel('data/新高度' + str(newh) + '风数据.xlsx')


computeWineDataAtNewAltitudeAndSave(originalWindData, 70)