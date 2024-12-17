# This is a sample Python script.
from weatherTypes import weatherTypes
import plotting
from metOcean import getMetOcean
from climateIndices import climateIndices

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import pickle
import sys
import subprocess

if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


savePath = '/volumes/macDrive/va/'
slpPath = '/volumes/macDrive/prmsl/'
wisPath = '/volumes/macDrive/WIS63142/'
wlPath = '/users/dylananderson/documents/data/frfWaterLevel/'
startTime = [1979, 2, 1]
endTime = [2024, 9, 30]
slpMemory = True
#
# climate = climateIndices(awtStart=1880,awtEnd=2024,savePath=savePath)
# climate.atlanticAWT(plotOutput=True)
# climate.mjo(historicalSimNum=100,futureSimNum=100,loadPrevious=False,plotOutput=True)
#
#
wts = weatherTypes(slpPath=slpPath,startTime=startTime,endTime=endTime,savePath=savePath, slpMemory=slpMemory,avgTime=12,resolution=1,
                   latBot=-5,latTop=65,lonRight=0,basin='atlantic')
# wts.extractCFSR(printToScreen=True,estelaMat='/volumes/anderson/ESTELA/out/NagsHead2/NagsHead2_obj.mat')
# #
wts.extractCFSR(loadPrior=True,loadPickle='/volumes/macDrive/va/slps12hr1degRes.pickle')



plotting.plotSlpExample(struct=wts)
# wts.pcaOfSlps()
wts.pcaOfSlps(loadPrior=True,loadPickle='/volumes/macDrive/va/pcas12hr1degRes.pickle')

plotting.plotEOFs(struct=wts)


metOcean = getMetOcean(wlPath=wlPath,wisPath=wisPath,startTime=startTime,endTime=endTime,shoreNormal=0)


stormPickle = '/users/dylananderson/Documents/projects/frf_python_share/stormHs95Over12Hours.pickle'
with open(stormPickle, "rb") as input_file:
    inputStorms = pickle.load(input_file)

metOcean.timeWave = inputStorms['combinedTimeWIS']
metOcean.Hs = inputStorms['combinedHsWIS']
metOcean.Tp = inputStorms['combinedTpWIS']
metOcean.Dm = inputStorms['combinedDmWIS']


# metOcean.getWISLocal()
# # metOcean.getWIS()
# # #def getWISThredds(self,basin,buoy,**kwargs):
# # metOcean.getWISThredds(basin = 'Atlantic',buoy = 'ST63218',variables = ['waveHs','waveTpPeak','waveMeanDirection'])
# # plotting.plotOceanConditions(struct=metOcean)


# wts.wtClusters(numClusters=64,minGroupSize=50,TCs=False,RG='seasonal',Basin=b'NA',alphaRG=0.2)
wts.wtClusters(numClusters=64,minGroupSize=30,TCs=False,RG='waves',Basin=b'NA',alphaRG=0.2,met=metOcean,loadPrior=True,
               loadPickle=os.path.join(savePath,'dwts12hr1degRes64withRG02.pickle'))
plotting.plotWTs(struct=wts,withTCs=False)


asdfg

plotting.plotSeasonal(struct=wts)

metOcean.getWaterLevels()


with open(os.path.join(savePath,'latestData.pickle'), "rb") as input_file:
   priorComputations = pickle.load(input_file)
# wts = priorComputations['wts']
# duckMET = priorComputations['duckMet']
climate = priorComputations['climate']

# #
wts.separateHistoricalHydrographs(metOcean=metOcean,loadPrior=True,loadPickle='/volumes/macDrive/va/hydros.pickle')
wts.metOceanCopulas(loadPrior=True,loadPickle='/volumes/macDrive/va/copulas.pickle')
# # #
wts.alrSimulations(climate=climate,historicalSimNum=100,futureSimNum=100,futureSimStart=2024,futureSimEnd=2124)#,
                   #loadPrior=True,loadPickle='/volumes/macDrive/va/simDwts.pickle')
# plotting.plotSeasonalValidation(wts,20,12)
# plotting.plotTotalProbabilityValidation(wts,20)
# wts.simsFutureInterpolated(simNum=100)

# with open('/volumes/macDrive/va/percentWindows.pickle', "rb") as input_file:
#    priorComps = pickle.load(input_file)
# # wts = priorComputations['wts']
# # duckMET = priorComputations['duckMet']
# percentWindows = priorComps['percentWindows']
# wts.simsHistoricalInterpolated(simNum=100,percentWindows=percentWindows)
# wts.simsFutureValidated(met=metOcean)

# import os
# import pickle
# outdict = {}
# outdict['metOcean'] = metOcean
# outdict['wts'] = wts
# outdict['climate'] = climate
# outdict['endTime'] = endTime
# outdict['startTime'] = startTime
# with open(os.path.join(savePath,'latestData.pickle'), 'wb') as f:
#     pickle.dump(outdict, f)



