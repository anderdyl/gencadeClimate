import xarray as xr
import os
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
import cftime
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import itertools
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression
import pickle
import random as rm
from functions import copulaSimulation
from itertools import groupby


class climateIndices():
    '''
    Class containing camera data and functions'''

    def __init__(self, **kwargs):

        self.awtStart = kwargs.get('awtStart', 1880)
        self.awtEnd = kwargs.get('awtEnd',2022)
        self.ersstFolder = kwargs.get('ersstFolder', "/users/dylananderson/Documents/data/ERSSTv5/")
        self.latTop = kwargs.get('latTop', 50)
        self.resolution = kwargs.get('resolution',1)
        self.avgTime = kwargs.get('avgTime',24)
        self.startTime = kwargs.get('startTime',[1979,1,1])
        self.endTime = kwargs.get('endTime',[2020,12,31])
        self.slpMemory = kwargs.get('slpMemory',False)
        self.slpPath = kwargs.get('slpPath')
        # self.lonLeft = kwargs.get('cameraID', 'c1')
        # self.lonRight = kwargs.get('rawPath')
        # self.latBottom = kwargs.get('nFrames', 1)
        # self.latTop = kwargs.get('startFrame', 0)

    def atlanticAWT(self,loadPrevious=False,plotOutput=False):


        if loadPrevious == True:

            print('need to know what variables to load in this space')

        else:
            data_folder="/users/dylananderson/Documents/data/ERSSTv5/"


            years = np.arange(self.awtStart,self.awtEnd)
            months = np.arange(1,13)
            ogTime = []
            for ii in years:
                for hh in months:
                    if hh < 10:
                        date = str(ii) + "0" + str(hh)
                    else:
                        date = str(ii) + str(hh)

                    file = "ersst.v5." + date + ".nc"
                    #print(file)
                    if ii == self.awtStart and hh < 6:
                        print("skipping {}/{}".format(ii,hh))
                    else:
                        if ii == self.awtStart and hh == 6:
                            with xr.open_dataset(os.path.join(data_folder, file)) as ds:
                                temp = ds
                                SSTvalues = ds['sst']
                                ogTime.append(datetime.datetime(ii,hh,1))
                        elif ii == (self.awtEnd-1) and hh > 5:
                            print("skipping {}/{}".format(ii,hh))
                        else:
                            with xr.open_dataset(os.path.join(data_folder,file)) as ds:
                                SSTvalues = xr.concat([SSTvalues,ds['sst']],dim="time")
                                ogTime.append(datetime.datetime(ii,hh,1))



            dt = datetime.datetime(self.awtStart, 6, 1)
            end = datetime.datetime((self.awtEnd-1), 6, 1)
            step = relativedelta(years=1)
            sstTime = []
            while dt < end:
                sstTime.append(dt)
                dt += step

            data = SSTvalues.squeeze("lev")

            # parse data to xr.Dataset
            xds_predictor = xr.Dataset(
                {
                    'SST': (('longitude','latitude','time'), data.data.T),
                },
                coords = {
                    'longitude': SSTvalues.lon.values,
                    'latitude': SSTvalues.lat.values,
                    'time': ogTime,
                }
            )


            var_name = "SST"
            y1 = self.awtStart
            y2 = self.awtEnd-1
            m1 = 6
            m2 = 5
            subset = xds_predictor.sel(longitude=slice(280,350),latitude=slice(0,65))


            d1 = datetime.datetime(self.awtStart, 6, 1)
            dt = datetime.datetime(self.awtStart, 6, 1)
            end = datetime.datetime((self.awtEnd-1), 6, 1)
            step = relativedelta(months=1)
            monthlyTime = []
            while dt < end:
                monthlyTime.append(dt)
                dt += step


            timeDelta = np.array([(d - d1).days/365.25 for d in monthlyTime])

            tempdata_runavg = np.nan*np.ones(subset["SST"].shape)

            for lon in subset.longitude.values:
                for lat in subset.latitude.values:
                    # indexes
                    ix_lon = np.where(subset.longitude == lon)
                    ix_lat = np.where(subset.latitude == lat)
                    data_pnt = subset["SST"].loc[lon, lat, :]
                    if ~np.any(np.isnan(data_pnt.values)):
                        model = LinearRegression()
                        X = np.reshape(timeDelta, (len(timeDelta), 1))
                        model.fit(X, data_pnt.values)
                        trend = model.predict(X)
                        detrended = [data_pnt.values[i] - trend[i] for i in range(0,len(data_pnt.values))]
                        tempdata_runavg[ix_lon,ix_lat,:] = detrended


            d1 = datetime.datetime(self.awtStart, 6, 1)
            dt = datetime.datetime(self.awtStart, 6, 1)
            end = datetime.datetime((self.awtEnd-1), 6, 1)
            step = relativedelta(years=1)
            annualTime = []
            while dt < end:
                annualTime.append(dt)
                dt += step

            Xs = subset.longitude.values
            Ys = subset.latitude.values
            [XR, YR] = np.meshgrid(Xs, Ys)

            # if plotOutput:
            #     plt.figure()
            #     p1 = plt.subplot2grid((1, 1), (0, 0))
            #     spatialField = tempdata_runavg[:,:,-1]#np.reshape(var_anom_mean.values,(33,36))
            #     m = Basemap(projection='merc', llcrnrlat=-40, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10, resolution='c')
            #     m.drawcoastlines()
            #     cx, cy = m(XR, YR)
            #     CS = m.contour(cx, cy, spatialField.T),# np.arange(0,0.023,.003), cmap=cm.RdBu_r, shading='gouraud')


            nlon,nlat,ntime = np.shape(tempdata_runavg)

            collapsed = np.reshape(tempdata_runavg,(nlon*nlat, ntime))

            annual = np.nan*np.ones((int(nlon*nlat),int(ntime/12)))
            c = 0
            for hh in range(int(ntime/12)):
                annual[:,hh] = np.nanmean(collapsed[:,c:c+12],axis=1)
                c = c + 12

            index = ~np.isnan(annual[:,0])
            badIndex = np.isnan(annual[:,0])
            ocean = [i for i, x in enumerate(index) if x]
            land = [i for i, x in enumerate(badIndex) if x]
            realDataAnoms = annual[index,:]

            var_anom_mean = np.nanmean(realDataAnoms.T,axis=0)
            var_anom_std = np.nanstd(realDataAnoms.T,axis=0)
            timeSeries_mean = np.nanmean(realDataAnoms,axis=0)

            nk_m = np.kron(np.ones(((y2 - y1), 1)), var_anom_mean)
            nk_s = np.kron(np.ones(((y2 - y1), 1)), var_anom_std)
            var_anom_demean = (realDataAnoms.T - nk_m) / nk_s
            ipca = PCA()
            PCs = ipca.fit_transform(var_anom_demean)

            EOFs = ipca.components_
            variance = ipca.explained_variance_
            nPercent = variance / np.sum(variance)
            APEV = np.cumsum(variance) / np.sum(variance) * 100.0
            nterm = np.where(APEV <= 0.95 * 100)[0][-1]

            PC1 = PCs[:, 0]
            PC2 = PCs[:, 1]
            PC3 = PCs[:, 2]
            normPC1 = np.divide(PC1, np.nanmax(PC1)) * nPercent[0]
            normPC2 = np.divide(PC2, np.nanmax(PC2)) * nPercent[1]
            normPC3 = np.divide(PC3, np.nanmax(PC3)) * nPercent[2]

            n_components = 3  # !!!!
            pcAggregates = np.full((len(normPC1), n_components), np.nan)
            pcAggregates[:, 0] = normPC1
            pcAggregates[:, 1] = normPC2
            pcAggregates[:, 2] = normPC3

            n_clusters = 6
            kmeans = KMeans(n_clusters, init='k-means++', random_state=100)  # 80
            data = pcAggregates
            data1 = data / np.std(data, axis=0)
            awt_bmus_og = kmeans.fit_predict(data1)
            # awt_bmus2 = awt_bmus
            awt_bmus2 = np.nan * np.ones((np.shape(awt_bmus_og)))

            avgSST = []
            for hh in np.unique(awt_bmus_og):
                indexAWT = np.where(awt_bmus_og == hh)
                avgSST.append(np.nanmean(annual[:, indexAWT[0]]))
                #print(np.nanmean(annual[:, indexAWT[0]]))

            order = np.argsort(np.asarray(avgSST))#[0, 4, 5, 3, 2, 1]

            print(order)

            for hh in np.arange(0, 6):
                indexOR = np.where(awt_bmus_og == order[hh])
                awt_bmus2[indexOR] = np.ones((len(indexOR[0], ))) * hh
            awt_bmus = awt_bmus2



            if plotOutput:
                plt.figure()
                gs2 = gridspec.GridSpec(2, 3)
                for hh in np.unique(awt_bmus):
                    indexAWT = np.where(awt_bmus2 == hh)
                    # rectField = np.nanmean(subset['SST'][:, :, indexAWT[0]], axis=2)
                    # rectField = np.nanmean(tempdata_runavg[:, :, indexAWT[0]], axis=2)
                    rectField = np.reshape(np.nanmean(annual[:, indexAWT[0]], axis=1), (36, 33))
                    ax = plt.subplot(gs2[int(hh)])
                    Xs = subset.longitude.values
                    Ys = subset.latitude.values
                    [XR, YR] = np.meshgrid(Xs, Ys)
                    m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
                                resolution='c')
                    m.drawcoastlines()
                    cx, cy = m(XR, YR)
                    CS = m.contour(cx, cy, rectField.T, np.arange(-0.8, 0.8, .05), cmap=cm.RdBu_r, shading='gouraud')
                    ax.set_title('AWT #{} = {} years'.format(int(hh), len(indexAWT[0])))

                plt.colorbar(CS, ax=ax)



            d1 = datetime.datetime(1979, 6, 1)
            dt = datetime.datetime(1979, 6, 1)
            end = datetime.datetime(2022, 6, 2)
            step = relativedelta(days=1)
            dailyTime = []
            while dt < end:
                dailyTime.append(dt)
                dt += step

            DailyDatesMatrix = np.array([[r.year, r.month, r.day] for r in dailyTime])

            dailyAWT = np.ones((len(dailyTime),))
            dailyPC1 = np.ones((len(dailyTime),))
            dailyPC2 = np.ones((len(dailyTime),))
            dailyPC3 = np.ones((len(dailyTime),))

            anIndex = np.where(np.array(annualTime) >= datetime.datetime(1979, 5, 31))
            subsetAnnualTime = np.array(annualTime)[anIndex]
            # subsetAnnualTime = np.array(annualTime)
            subsetAWT = awt_bmus2[anIndex]
            # subsetPCs = pcAggregates[anIndex[0],:]#PCs[anIndex,:]
            subsetPCs = PCs[anIndex[0], :]

            for i in range(len(subsetAWT)):
                sSeason = np.where((DailyDatesMatrix[:, 0] == subsetAnnualTime[i].year) & (
                            DailyDatesMatrix[:, 1] == subsetAnnualTime[i].month) & (DailyDatesMatrix[:, 2] == 1))
                ssSeason = np.where((DailyDatesMatrix[:, 0] == subsetAnnualTime[i].year + 1) & (
                            DailyDatesMatrix[:, 1] == subsetAnnualTime[i].month) & (DailyDatesMatrix[:, 2] == 1))

                dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1] = subsetAWT[i] * dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]
                dailyPC1[sSeason[0][0]:ssSeason[0][0] + 1] = subsetPCs[i, 0] * np.ones(
                    len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
                dailyPC2[sSeason[0][0]:ssSeason[0][0] + 1] = subsetPCs[i, 1] * np.ones(
                    len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )
                dailyPC3[sSeason[0][0]:ssSeason[0][0] + 1] = subsetPCs[i, 2] * np.ones(
                    len(dailyAWT[sSeason[0][0]:ssSeason[0][0] + 1]), )

            # make a markov chain of the AWT clusters

            chain = {}
            n_words = len(awt_bmus)
            for i, key1 in enumerate(awt_bmus):
                if n_words > i + 2:
                    key2 = awt_bmus[i + 1]
                    word = awt_bmus[i + 2]
                    if (key1, key2) not in chain:
                        chain[(key1, key2)] = [word]
                    else:
                        chain[(key1, key2)].append(word)

            print('Chain size: {0} distinct bmu pairs.'.format(len(chain)))
            #
            # chain3 = {}
            # n_words = len(awt_bmus)
            # for i, key1 in enumerate(awt_bmus):
            #     if n_words > i + 3:
            #         key2 = awt_bmus[i + 1]
            #         key3 = awt_bmus[i + 2]
            #         word = awt_bmus[i + 3]
            #         if (key1, key2, key3) not in chain3:
            #             chain3[(key1, key2, key3)] = [word]
            #         else:
            #             chain3[(key1, key2, key3)].append(word)
            # print('Chain size: {0} distinct bmu pairs.'.format(len(chain3)))
            print(chain)
            sim_num = 100
            sim_years = 500
            evbmus_sim = np.nan * np.ones((sim_num, (sim_years)))
            key = (awt_bmus[-2], awt_bmus[-1])
            for gg in range(sim_num):
                bmu_sim = [awt_bmus[-2], awt_bmus[-1]]
                c = 2
                while len(bmu_sim) < (sim_years):
                    w = rm.choice(chain[key])
                    bmu_sim.append(w)
                    key = (key[1], w)
                    c = c + 1
                evbmus_sim[gg, :] = bmu_sim

            # sim_num = 100
            bmus = awt_bmus  # [1:]
            evbmus_sim = evbmus_sim  # evbmus_simALR.T

            # Lets make a plot comparing probabilities in sim vs. historical
            probH = np.nan * np.ones((n_clusters,))
            probS = np.nan * np.ones((sim_num, n_clusters))
            for h in np.unique(bmus):
                findH = np.where((bmus == h))[0][:]
                probH[int(h - 1)] = len(findH) / len(bmus)

                for s in range(sim_num):
                    findS = np.where((evbmus_sim[s, :] == h))[0][:]
                    probS[s, int(h - 1)] = len(findS) / len(evbmus_sim[s, :])

            # from alrPlotting import colors_mjo
            # from alrPlotting import colors_awt
            etcolors = cm.jet(np.linspace(0, 1, 24))  # 70-20))
            tccolors = np.flipud(cm.autumn(np.linspace(0, 1, 2)))  # 21)))
            dwtcolors = np.vstack((etcolors, tccolors[1:, :]))
            # dwtcolors = colors_mjo()

            if plotOutput:
                plt.figure()
                ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
                tempPs = np.nan * np.ones((6,))
                for i in range(6):
                    temp = probS[:, i]
                    temp2 = probH[i]
                    box1 = ax.boxplot(temp, positions=[temp2], widths=.01, notch=True, patch_artist=True, showfliers=False)
                    plt.setp(box1['boxes'], color=dwtcolors[i])
                    plt.setp(box1['means'], color=dwtcolors[i])
                    plt.setp(box1['fliers'], color=dwtcolors[i])
                    plt.setp(box1['whiskers'], color=dwtcolors[i])
                    plt.setp(box1['caps'], color=dwtcolors[i])
                    plt.setp(box1['medians'], color=dwtcolors[i], linewidth=0)
                    tempPs[i] = np.mean(temp)
                    # box1['boxes'].set(facecolor=dwtcolors[i])
                    # plt.set(box1['fliers'],markeredgecolor=dwtcolors[i])
                ax.plot([0, 0.3], [0, 0.3], 'k--', zorder=10)
                plt.xlim([0, 0.3])
                plt.ylim([0, 0.3])
                plt.xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3], ['0', '0.05', '0.10', '0.15', '0.20', '0.25', '0.3'])
                plt.xlabel('Historical Probability')
                plt.ylabel('Simulated Probability')
                plt.title('Validation of ALR SWT Simulations')


            a = list(bmus)
            seq = list()
            for i in np.arange(1, 7):
                temp = [len(list(v)) for k, v in groupby(a) if k == i - 1]
                seq.append(temp)

            simseqPers = list()
            for hhh in range(sim_num):
                b = list(evbmus_sim[hhh, :])
                seq_sim = list()
                for i in np.arange(1, 7):
                    temp2 = [len(list(v)) for k, v in groupby(b) if k == i - 1]
                    seq_sim.append(temp2)
                simseqPers.append(seq_sim)

            persistReal = np.nan * np.ones((6, 5))
            for dwt in np.arange(1, 7):
                sortDurs = np.sort(seq[dwt - 1])
                realPercent = np.nan * np.ones((5,))
                for qq in np.arange(1, 6):
                    realInd = np.where((sortDurs <= qq))
                    realPercent[qq - 1] = len(realInd[0]) / len(sortDurs)
                persistReal[dwt - 1, :] = realPercent

            persistSim = list()
            for dwt in np.arange(1, 7):
                persistDWT = np.nan * np.ones((sim_num, 5))
                for simInd in range(sim_num):

                    sortDursSim = np.sort(simseqPers[simInd][dwt - 1])
                    simPercent = np.nan * np.ones((5,))
                    for qq in np.arange(1, 6):
                        simIndex = np.where((sortDursSim <= qq))
                        simPercent[qq - 1] = len(simIndex[0]) / len(sortDursSim)
                    persistDWT[simInd, :] = simPercent
                persistSim.append(persistDWT)

            x = [0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5.5]
            if plotOutput:
                plt.figure()
                gs2 = gridspec.GridSpec(2, 3)
                for xx in range(6):
                    ax = plt.subplot(gs2[xx])
                    ax.boxplot(persistSim[xx])
                    y = [persistReal[xx, 0], persistReal[xx, 0], persistReal[xx, 1], persistReal[xx, 1], persistReal[xx, 2],
                         persistReal[xx, 2], persistReal[xx, 3], persistReal[xx, 3], persistReal[xx, 4],
                         persistReal[xx, 4], ]
                    ax.plot(x, y, color=dwtcolors[xx])
                    ax.set_ylim([0.25, 1.05])

            copulaData = list()
            for i in range(len(np.unique(bmus))):

                tempInd = np.where(((bmus) == i))
                dataCop = []
                for kk in range(len(tempInd[0])):
                    dataCop.append(list([PC1[tempInd[0][kk]], PC2[tempInd[0][kk]], PC3[tempInd[0][kk]]]))
                copulaData.append(dataCop)

            gevCopulaSims = list()
            for i in range(len(np.unique(bmus))):
                tempCopula = np.asarray(copulaData[i])
                kernels = ['KDE', 'KDE', 'KDE']
                samples = copulaSimulation(tempCopula, kernels, 100000)
                print('generating samples for AWT {}'.format(i))
                gevCopulaSims.append(samples)

            # convert synthetic markovs to PC values
            # Fill in the Markov chain bmus with RMM vales
            pc1Sims = list()
            pc2Sims = list()
            pc3Sims = list()
            for kk in range(sim_num):
                tempSimulation = evbmus_sim[kk, :]
                tempPC1 = np.nan * np.ones((np.shape(tempSimulation)))
                tempPC2 = np.nan * np.ones((np.shape(tempSimulation)))
                tempPC3 = np.nan * np.ones((np.shape(tempSimulation)))

                groups = [list(j) for i, j in groupby(tempSimulation)]
                c = 0
                for gg in range(len(groups)):
                    getInds = rm.sample(range(1, 100000), len(groups[gg]))
                    tempPC1s = gevCopulaSims[int(groups[gg][0])][getInds[0], 0]
                    tempPC2s = gevCopulaSims[int(groups[gg][0])][getInds[0], 1]
                    tempPC3s = gevCopulaSims[int(groups[gg][0])][getInds[0], 2]
                    tempPC1[c:c + len(groups[gg])] = tempPC1s
                    tempPC2[c:c + len(groups[gg])] = tempPC2s
                    tempPC3[c:c + len(groups[gg])] = tempPC3s
                    c = c + len(groups[gg])
                pc1Sims.append(tempPC1)
                pc2Sims.append(tempPC2)
                pc3Sims.append(tempPC3)

            # sim_years = 100
            # start simulation at PCs available data
            d1 = datetime.datetime(2022, 6, 1)
            d2 = datetime.datetime(d1.year + sim_years, d1.month, d1.day)
            dates_sim2 = [d1 + datetime.timedelta(days=i) for i in range((d2 - d1).days + 1)]
            # dates_sim = dates_sim[0:-1]

            # sim_years = 100
            # start simulation at PCs available data
            d1 = datetime.datetime(2022, 6, 1)  # x2d(xds_cov_fit.time[0])
            d2 = datetime.datetime(2022 + int(sim_years), 6, 1)  # datetime(d1.year+sim_years, d1.month, d1.day)
            dt = datetime.date(2022, 6, 1)
            end = datetime.date(2022 + int(sim_years), 7, 1)
            # step = datetime.timedelta(months=1)
            step = relativedelta(months=1)
            dates_sim = []
            while dt < end:
                dates_sim.append(dt)  # .strftime('%Y-%m-%d'))
                dt += step


            self.pc1Sims = pc1Sims
            self.pc2Sims = pc2Sims
            self.pc3Sims = pc3Sims
            self.evbmus_sim = evbmus_sim
            self.dates_sim = dates_sim

            # samplesPickle = 'awtSimulations.pickle'
            # outputSamples = {}
            # outputSamples['pc1Sims'] = pc1Sims
            # outputSamples['pc2Sims'] = pc2Sims
            # outputSamples['pc3Sims'] = pc3Sims
            # # outputSamples['pc4Sims'] = pc4Sims
            # outputSamples['evbmus_sim'] = evbmus_sim
            # outputSamples['dates_sim'] = dates_sim
            # with open(samplesPickle, 'wb') as f:
            #     pickle.dump(outputSamples, f)

            # awtPickle = 'awtPCs.pickle'
            # outputMWTs = {}
            # outputMWTs['PC1'] = PC1
            # outputMWTs['PC2'] = PC2
            # outputMWTs['PC3'] = PC3
            # outputMWTs['normPC1'] = normPC1
            # outputMWTs['normPC2'] = normPC2
            # outputMWTs['normPC3'] = normPC3
            # outputMWTs['awt_bmus'] = awt_bmus
            # outputMWTs['annualTime'] = annualTime
            # outputMWTs['dailyAWT'] = dailyAWT
            # outputMWTs['dailyDates'] = DailyDatesMatrix
            # outputMWTs['dailyTime'] = dailyTime
            # outputMWTs['dailyPC1'] = dailyPC1
            # outputMWTs['dailyPC2'] = dailyPC2
            # outputMWTs['dailyPC3'] = dailyPC3
            # outputMWTs['nPercent'] = nPercent
            # with open(awtPickle, 'wb') as f:
            #     pickle.dump(outputMWTs, f)

            #
            # mwtPickle = 'sstWTsPCsAndAllData.pickle'
            # outputMWTs = {}
            # outputMWTs['PCs'] = PCs
            # outputMWTs['EOFs'] = EOFs
            # outputMWTs['nPercent'] = nPercent
            # outputMWTs['awt_bmus'] = awt_bmus
            # outputMWTs['n_components'] = n_components
            # outputMWTs['variance'] = variance
            # outputMWTs['ocean'] = ocean
            # outputMWTs['land'] = land
            # outputMWTs['realDataAnoms'] = realDataAnoms
            # outputMWTs['tempdata_runavg'] = tempdata_runavg
            # outputMWTs['collapsed'] = collapsed
            # outputMWTs['annual'] = annual
            # outputMWTs['annualTime'] = annualTime
            # outputMWTs['subset'] = subset
            # outputMWTs['data'] = data
            #
            # with open(mwtPickle,'wb') as f:
            #     pickle.dump(outputMWTs, f)


    def mjo(self,loadPrevious=False,plotOutput=False):


        if loadPrevious == True:

            print('need to know what variables to load in this space')

        else:
            data_folder="/users/dylananderson/Documents/data/ERSSTv5/"