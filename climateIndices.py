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
        self.awtEnd = kwargs.get('awtEnd',2024)
        self.ersstFolder = kwargs.get('ersstFolder', "/users/dylananderson/Documents/data/ERSSTv5/")
        self.latTop = kwargs.get('latTop', 50)
        self.resolution = kwargs.get('resolution',1)
        self.avgTime = kwargs.get('avgTime',24)
        self.startTime = kwargs.get('startTime',[1979,1,1])
        self.endTime = kwargs.get('endTime',[2024,5,31])
        self.slpMemory = kwargs.get('slpMemory',False)
        self.slpPath = kwargs.get('slpPath')
        self.lonLeft = kwargs.get('lonLeft', 280)
        self.lonRight = kwargs.get('lonRight',350)
        self.latBottom = kwargs.get('latBottom', 0)
        self.latTop = kwargs.get('latTop', 65)

    def atlanticAWT(self,loadPrevious=False,plotOutput=False):


        if loadPrevious == True:

            print('need to know what variables to load in this space')

        else:
            data_folder="/users/dylananderson/Documents/data/ERSSTv5/"


            years = np.arange(self.awtStart,self.awtEnd+1)
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
                        elif ii == (self.awtEnd) and hh > 5:
                            print("skipping {}/{}".format(ii,hh))
                        else:
                            with xr.open_dataset(os.path.join(data_folder,file)) as ds:
                                SSTvalues = xr.concat([SSTvalues,ds['sst']],dim="time")
                                ogTime.append(datetime.datetime(ii,hh,1))



            dt = datetime.datetime(self.awtStart, 6, 1)
            end = datetime.datetime((self.awtEnd), 6, 1)
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
            y2 = self.awtEnd
            m1 = 6
            m2 = 5
            subset = xds_predictor.sel(longitude=slice(self.lonLeft,self.lonRight),latitude=slice(self.latBottom,self.latTop))


            d1 = datetime.datetime(self.awtStart, 6, 1)
            dt = datetime.datetime(self.awtStart, 6, 1)
            end = datetime.datetime((self.awtEnd), 6, 1)
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
            end = datetime.datetime((self.awtEnd), 6, 1)
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
                    ax = plt.subplot(gs2[int(hh)])
                    Xs = subset.longitude.values
                    Ys = subset.latitude.values
                    [XR, YR] = np.meshgrid(Xs, Ys)
                    # rectField = np.reshape(np.nanmean(annual[:, indexAWT[0]], axis=1), (int((self.lonRight-self.lonRight)/2+1), ))
                    rectField = np.reshape(np.nanmean(annual[:, indexAWT[0]], axis=1), (len(Xs),len(Ys)))

                    m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
                                resolution='c')
                    m.drawcoastlines()
                    cx, cy = m(XR, YR)
                    CS = m.contour(cx, cy, rectField.T, np.arange(-0.8, 0.8, .05), cmap=cm.RdBu_r, shading='gouraud')
                    ax.set_title('AWT #{} = {} years'.format(int(hh), len(indexAWT[0])))

                plt.colorbar(CS, ax=ax)



            d1 = datetime.datetime(1979, 6, 1)
            dt = datetime.datetime(1979, 6, 1)
            end = datetime.datetime(self.awtEnd, 6, 2)
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

            self.dailyPC1 = dailyPC1[0:-1]
            self.dailyPC2 = dailyPC2[0:-1]
            self.dailyPC3 = dailyPC3[0:-1]

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
            self.awtBmusSim = evbmus_sim
            self.awtDatesSim = dates_sim

            import pickle
            samplesPickle = 'awtSimulations.pickle'
            outputSamples = {}
            outputSamples['pc1Sims'] = pc1Sims
            outputSamples['pc2Sims'] = pc2Sims
            outputSamples['pc3Sims'] = pc3Sims
            outputSamples['evbmus_sim'] = evbmus_sim
            outputSamples['dates_sim'] = dates_sim
            outputSamples['PC1'] = PC1
            outputSamples['PC2'] = PC2
            outputSamples['PC3'] = PC3
            outputSamples['normPC1'] = normPC1
            outputSamples['normPC2'] = normPC2
            outputSamples['normPC3'] = normPC3
            outputSamples['awt_bmus'] = awt_bmus
            outputSamples['annualTime'] = annualTime
            outputSamples['dailyAWT'] = dailyAWT
            outputSamples['dailyDates'] = DailyDatesMatrix
            outputSamples['dailyTime'] = dailyTime
            outputSamples['dailyPC1'] = dailyPC1
            outputSamples['dailyPC2'] = dailyPC2
            outputSamples['dailyPC3'] = dailyPC3
            outputSamples['nPercent'] = nPercent

            with open(samplesPickle, 'wb') as f:
                pickle.dump(outputSamples, f)

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


    def mjo(self,historicalSimNum,futureSimNum,loadPrevious=False,plotOutput=False):


        if loadPrevious == True:

            print('need to know what variables to load in this space')

        else:
            import datetime
            from dateutil.relativedelta import relativedelta
            import numpy as np
            import pandas as pd
            import csv
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            import matplotlib.colors as mcolors
            import matplotlib.patches as patches
            from functions import ALR_WRP
            from functions import xds_common_dates_daily as xcd_daily
            from functions import xds_reindex_daily as xr_daily

            _faspect = 1.618
            _fsize = 9.8
            _fdpi = 128

            year = []
            month = []
            day = []
            RMM1 = []
            RMM2 = []
            phase = []
            amplitude = []

            #http://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt

            from urllib import request
            remote_url = 'http://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt'
            opener = request.build_opener()
            opener.addheaders = [('User-Agent', 'MyApp/1.0')]
            request.install_opener(opener)
            file = 'mjo.txt'
            request.urlretrieve(remote_url, file)
            # Missing Value= 1.E36 or 999
            with open('mjo.txt', newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='\t')  # , quotechar='|')
                # line1 = txtfile.readline()
                # line2 = txtfile.readline()
                # data = txtfile.readlines()
                next(csvreader)
                next(csvreader)
                for row in csvreader:
                    print('{}'.format(row[0]))
                    temp = row[0].split()
                    year.append(int(temp[0]))
                    month.append(int(temp[1]))
                    day.append(int(temp[2]))
                    RMM1.append(float(temp[3]))
                    RMM2.append(float(temp[4]))
                    phase.append(int(temp[5]))
                    amplitude.append(float(temp[6]))

            mjoPhase = np.asarray(phase)
            mjoRmm1 = np.asarray(RMM1)
            mjoRmm2 = np.asarray(RMM2)
            dt = datetime.date(year[0], month[0], day[0])
            end = datetime.date(year[-1], month[-1], day[-1])
            step = relativedelta(days=1)
            mjoTime = []
            while dt < end:
                mjoTime.append(dt)  # .strftime('%Y-%m-%d'))
                dt += step


            index = np.where((np.asarray(mjoTime) >= datetime.date(1979,6,1)) & (np.asarray(mjoTime) < datetime.date(self.awtEnd,6,1)))

            mjoRmm1 = mjoRmm1[index]
            mjoRmm2 = mjoRmm2[index]
            mjoPhase = mjoPhase[index]

            def MJO_Categories(rmm1, rmm2, phase):
                '''
                Divides MJO data in 25 categories.

                rmm1, rmm2, phase - MJO parameters

                returns array with categories time series
                and corresponding rmm
                '''

                rmm = np.sqrt(rmm1 ** 2 + rmm2 ** 2)
                categ = np.empty(rmm.shape) * np.nan

                for i in range(1, 9):
                    s = np.squeeze(np.where(phase == i))
                    rmm_p = rmm[s]

                    # categories
                    categ_p = np.empty(rmm_p.shape) * np.nan
                    categ_p[rmm_p <= 1] = 25
                    categ_p[rmm_p > 1] = i + 8 * 2
                    categ_p[rmm_p > 1.5] = i + 8
                    categ_p[rmm_p > 2.5] = i
                    categ[s] = categ_p

                # get rmm_categ
                rmm_categ = {}
                for i in range(1, 26):
                    s = np.squeeze(np.where(categ == i))
                    rmm_categ['cat_{0}'.format(i)] = np.column_stack((rmm1[s], rmm2[s]))

                return categ.astype(int), rmm_categ

            def Plot_MJO_phases(rmm1, rmm2, phase, show=True):
                'Plot MJO data separated by phase'

                # parameters for custom plot
                size_points = 0.2
                size_lines = 0.8
                l_colors_phase = np.array(
                    [
                        [1, 0, 0],
                        [0.6602, 0.6602, 0.6602],
                        [1.0, 0.4961, 0.3125],
                        [0, 1, 0],
                        [0.2539, 0.4102, 0.8789],
                        [0, 1, 1],
                        [1, 0.8398, 0],
                        [0.2930, 0, 0.5078]
                    ]
                )

                color_lines_1 = (0.4102, 0.4102, 0.4102)

                # plot figure
                fig, ax = plt.subplots(1, 1, figsize=(_fsize, _fsize))
                ax.scatter(rmm1, rmm2, c='b', s=size_points)

                # plot data by phases
                for i in range(1, 9):
                    ax.scatter(
                        rmm1.where(phase == i),
                        rmm2.where(phase == i),
                        c=np.array([l_colors_phase[i - 1]]),
                        s=size_points)

                # plot sectors
                ax.plot([-4, 4], [-4, 4], color='k', linewidth=size_lines)
                ax.plot([-4, 4], [4, -4], color='k', linewidth=size_lines)
                ax.plot([-4, 4], [0, 0], color='k', linewidth=size_lines)
                ax.plot([0, 0], [-4, 4], color='k', linewidth=size_lines)

                # axis
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
                plt.xlabel('RMM1')
                plt.ylabel('RMM2')
                ax.set_aspect('equal')

                # show and return figure
                if show: plt.show()
                return fig

            # xds = xr.open_dataset(self.paths.site.MJO.hist)

            mjoBmus, mjoGroups = MJO_Categories(mjoRmm1, mjoRmm2, mjoPhase)

            rm1 = pd.DataFrame(mjoRmm1)
            rm2 = pd.DataFrame(mjoRmm2)
            ph = pd.DataFrame(mjoPhase)
            bmus = pd.DataFrame(mjoBmus)
            if plotOutput:
                axMJO = Plot_MJO_phases(rm1, rm2, ph)

            def colors_mjo():
                'custom colors for MJO 25 categories'

                l_named_colors = [
                    'lightskyblue', 'deepskyblue', 'royalblue', 'mediumblue',
                    'darkblue', 'darkblue', 'darkturquoise', 'turquoise',
                    'maroon', 'saddlebrown', 'chocolate', 'gold', 'orange',
                    'orangered', 'red', 'firebrick', 'Purple', 'darkorchid',
                    'mediumorchid', 'magenta', 'mediumslateblue', 'blueviolet',
                    'darkslateblue', 'indigo', 'darkgray',
                ]

                # get rgb colors as numpy array
                np_colors_rgb = np.array(
                    [mcolors.to_rgb(c) for c in l_named_colors]
                )

                return np_colors_rgb

            np_colors_rgb_categ = colors_mjo()

            def Plot_MJO_Categories(rmm1, rmm2, categ, show=True):
                'Plot MJO data separated by 25 categories'

                # parameters for custom plot
                size_lines = 0.8
                color_lines_1 = (0.4102, 0.4102, 0.4102)

                #  custom colors for mjo 25 categories
                np_colors_rgb_categ = colors_mjo()

                # plot figure
                fig, ax = plt.subplots(1, 1, figsize=(_fsize, _fsize))

                # plot sectors
                ax.plot([-4, 4], [-4, 4], color='k', linewidth=size_lines, zorder=9)
                ax.plot([-4, 4], [4, -4], color='k', linewidth=size_lines, zorder=9)
                ax.plot([-4, 4], [0, 0], color='k', linewidth=size_lines, zorder=9)
                ax.plot([0, 0], [-4, 4], color='k', linewidth=size_lines, zorder=9)

                # plot circles
                R = [1, 1.5, 2.5]

                for rr in R:
                    ax.add_patch(
                        patches.Circle(
                            (0, 0),
                            rr,
                            color='k',
                            linewidth=size_lines,
                            fill=False,
                            zorder=9)
                    )
                ax.add_patch(
                    patches.Circle((0, 0), R[0], fc='w', fill=True, zorder=10))

                # plot data by categories
                for i in range(1, 25):
                    if i > 8:
                        size_points = 0.2
                    else:
                        size_points = 1.7

                    ax.scatter(
                        rmm1.where(categ == i),
                        rmm2.where(categ == i),
                        c=[np_colors_rgb_categ[i - 1]],
                        s=size_points
                    )

                #  last category on top (zorder)
                ax.scatter(
                    rmm1.where(categ == 25),
                    rmm2.where(categ == 25),
                    c=[np_colors_rgb_categ[-1]],
                    s=0.2,
                    zorder=12
                )

                # TODO: category number
                rr = 0.3
                ru = 0.2
                l_pn = [
                    (-3, -1.5, '1'),
                    (-1.5, -3, '2'),
                    (1.5 - rr, -3, '3'),
                    (3 - rr, -1.5, '4'),
                    (3 - rr, 1.5 - ru, '5'),
                    (1.5 - rr, 3 - ru, '6'),
                    (-1.5, 3 - ru, '7'),
                    (-3, 1.5 - ru, '8'),
                    (-2, -1, '9'),
                    (-1, -2, '10'),
                    (1 - rr, -2, '11'),
                    (2 - rr, -1, '12'),
                    (2 - rr, 1 - ru, '13'),
                    (1 - rr, 2 - ru, '14'),
                    (-1, 2 - ru, '15'),
                    (-2, 1 - ru, '16'),
                    (-1.3, -0.6, '17'),
                    (-0.6, -1.3, '18'),
                    (0.6 - rr, -1.3, '19'),
                    (1.3 - rr, -0.6, '20'),
                    (1.3 - rr, 0.6 - ru, '21'),
                    (0.6 - rr, 1.3 - ru, '22'),
                    (-0.6, 1.3 - ru, '23'),
                    (-1.3, 0.6 - ru, '24'),
                    (0 - rr / 2, 0 - ru / 2, '25'),
                ]
                for xt, yt, tt in l_pn:
                    ax.text(xt, yt, tt, fontsize=15, fontweight='bold', zorder=11)

                # axis
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
                plt.xlabel('RMM1')
                plt.ylabel('RMM2')
                ax.set_aspect('equal')

                # show and return figure
                if show: plt.show()
                return fig

            if plotOutput:
                axMJO2 = Plot_MJO_Categories(rm1, rm2, bmus)

            def ClusterProbabilities(series, set_values):
                'return series probabilities for each item at set_values'

                us, cs = np.unique(series, return_counts=True)
                d_count = dict(zip(us, cs))

                # cluster probabilities
                cprobs = np.zeros((len(set_values)))
                for i, c in enumerate(set_values):
                    cprobs[i] = 1.0 * d_count[c] / len(series) if c in d_count.keys() else 0.0

                return cprobs

            def axplot_WT_Probs(ax, wt_probs,
                                ttl='', vmin=0, vmax=0.1,
                                cmap='Blues', caxis='black'):
                'axes plot WT cluster probabilities'

                # clsuter transition plot
                pc = ax.pcolor(
                    np.flipud(wt_probs),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors='k',
                )

                # customize axes
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(ttl, {'fontsize': 10, 'fontweight': 'bold'})

                # axis color
                plt.setp(ax.spines.values(), color=caxis)
                plt.setp(
                    [ax.get_xticklines(), ax.get_yticklines()],
                    color=caxis,
                )

                # axis linewidth
                if caxis != 'black':
                    plt.setp(ax.spines.values(), linewidth=3)

                return pc

            mjoTime = np.asarray(mjoTime)[index]
            mjoYear = np.asarray(year)[index]
            mjoDay = np.asarray(day)[index]
            mjoMonth = np.asarray(month)[index]
            self.mjoBmus = mjoBmus
            self.mjoGroups = mjoGroups
            self.mjoPhase = mjoPhase
            self.mjoRmm1 = mjoRmm1
            self.mjoRmm2 = mjoRmm2
            self.mjoTime = mjoTime
            self.index = index
            self.mjoYear = mjoYear
            self.mjoDay = mjoDay
            self.mjoMonth = mjoMonth

            self.copulaData = list()
            for i in range(len(np.unique(mjoBmus))):

                tempInd = np.where(((mjoBmus) == i+1))
                dataCop = []
                for kk in range(len(tempInd[0])):
                    dataCop.append(list([mjoRmm1[tempInd[0][kk]], mjoRmm2[tempInd[0][kk]]]))
                self.copulaData.append(dataCop)

            self.gevCopulaSims = list()
            for i in range(len(np.unique(mjoBmus))):
                tempCopula = np.asarray(self.copulaData[i])
                kernels = ['KDE', 'KDE']
                samples = copulaSimulation(tempCopula, kernels, 100000)
                print('generating samples for MJO {}'.format(i))
                self.gevCopulaSims.append(samples)
            # xds_MJO_fit = xr.Dataset(
            #     {
            #         'rmm1': (('time',), mjoRmm1),
            #         'rmm2': (('time',), mjoRmm2),
            #     },
            #     coords={'time': [datetime.datetime(mjoYear[r], mjoMonth[r], mjoDay[r]) for r in range(len(mjoDay))]}
            # )
            # # reindex to daily data after 1979-01-01 (avoid NaN)
            # xds_MJO_fit = xr_daily(xds_MJO_fit, datetime.datetime(1979, 6, 1))

            from datetime import datetime, timedelta
            self.xds_KMA_fit = xr.Dataset(
                {
                    'bmus': (('time',), mjoBmus),
                },
                coords={'time': [datetime(mjoYear[r], mjoMonth[r], mjoDay[r]) for r in range(len(mjoDay))]}
            )

            # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
            self.xds_PCs_fit = xr.Dataset(
                {
                    'PC1': (('time',), self.dailyPC1),
                    'PC2': (('time',), self.dailyPC3),
                    'PC3': (('time',), self.dailyPC3),
                },
                coords={'time': [datetime(mjoYear[r], mjoMonth[r], mjoDay[r]) for r in range(len(mjoDay))]}
            )
            # reindex annual data to daily data
            # xds_PCs_fit = xr_daily(xds_PCs_fit)

            # --------------------------------------
            # Mount covariates matrix

            # available data:
            # model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
            # model sim: xds_MJO_sim, xds_PCs_sim

            # covariates: FIT
            # d_covars_fit = xcd_daily([xds_MJO_fit, xds_PCs_fit, xds_KMA_fit])
            d_covars_fit = xcd_daily([self.xds_PCs_fit, self.xds_KMA_fit])

            # PCs covar
            cov_PCs = self.xds_PCs_fit.sel(time=slice(d_covars_fit[0], d_covars_fit[-1]))
            cov_1 = cov_PCs.PC1.values.reshape(-1, 1)
            cov_2 = cov_PCs.PC2.values.reshape(-1, 1)
            cov_3 = cov_PCs.PC3.values.reshape(-1, 1)

            # MJO covars
            # cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0], d_covars_fit[-1]))
            # cov_4 = cov_MJO.rmm1.values.reshape(-1, 1)
            # cov_5 = cov_MJO.rmm2.values.reshape(-1, 1)

            # join covars and norm.
            cov_T = np.hstack((cov_1, cov_2, cov_3))

            # generate xarray.Dataset
            cov_names = ['PC1', 'PC2', 'PC3']
            self.xds_cov_fit = xr.Dataset(
                {
                    'cov_values': (('time', 'cov_names'), cov_T),
                },
                coords={
                    'time': d_covars_fit,
                    'cov_names': cov_names,
                }
            )

            # use bmus inside covariate time frame
            self.xds_bmus_fit = self.xds_KMA_fit.sel(
                time=slice(d_covars_fit[0], d_covars_fit[-1])
            )

            bmus = self.xds_bmus_fit.bmus

            # Autoregressive logistic wrapper
            num_clusters = 25
            sim_num = 100
            fit_and_save = True  # False for loading
            p_test_ALR = '/Users/dylananderson/Documents/duneLifeCycles/'

            # ALR terms
            self.d_terms_settings = {
                'mk_order': 2,
                'constant': True,
                'long_term': False,
                'seasonality': (False,),  # [2, 4, 6]),
                'covariates': (True, self.xds_cov_fit),
            }
            # Autoregressive logistic wrapper
            ALRW = ALR_WRP(p_test_ALR)
            ALRW.SetFitData(
                num_clusters, self.xds_bmus_fit, self.d_terms_settings)

            ALRW.FitModel(max_iter=20000)

            # p_report = op.join(p_data, 'r_{0}'.format(name_test))

            ALRW.Report_Fit()  # '/media/dylananderson/Elements/NC_climate/testALR/r_Test', terms_fit==False)

            samplesPickle = 'mjoSimulations.pickle'
            outputSamples = {}
            outputSamples['mjoBmus'] = self.mjoBmus
            outputSamples['mjoGroups'] = self.mjoGroups
            outputSamples['mjoPhase'] = self.mjoPhase
            outputSamples['mjoRmm1'] = self.mjoRmm1
            outputSamples['mjoRmm2'] = self.mjoRmm2
            outputSamples['mjoTime'] = self.mjoTime
            outputSamples['index'] = self.index
            outputSamples['mjoYear'] = self.mjoYear
            outputSamples['mjoDay'] = self.mjoDay
            outputSamples['mjoMonth'] = self.mjoMonth

            if historicalSimNum > 0:
                # Historical Simulation
                # start simulation at PCs available data
                d1 = datetime(1979, 6, 1)  # x2d(xds_cov_fit.time[0])
                d2 = datetime(2023, 6, 1)  # datetime(d1.year+sim_years, d1.month, d1.day)
                dates_sim = [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]
                # print some info
                # print('ALR model fit   : {0} --- {1}'.format(
                #    d_covars_fit[0], d_covars_fit[-1]))
                print('ALR model sim   : {0} --- {1}'.format(
                    dates_sim[0], dates_sim[-1]))

                #  launch simulation
                xds_ALR = ALRW.Simulate(
                    historicalSimNum, dates_sim[0:-1], self.xds_cov_fit)

                self.historicalDatesSim = dates_sim

                # Save results for matlab plot
                self.historicalBmusSim = xds_ALR.evbmus_sims.values
                # evbmus_probcum = xds_ALR.evbmus_probcum.values


                # convert synthetic markovs to PC values
                # Fill in the Markov chain bmus with RMM vales
                self.rmm1Sims = list()
                self.rmm2Sims = list()
                for kk in range(historicalSimNum):
                    tempSimulation = self.historicalBmusSim[:,kk]
                    tempPC1 = np.nan * np.ones((np.shape(tempSimulation)))
                    tempPC2 = np.nan * np.ones((np.shape(tempSimulation)))

                    self.groups = [list(j) for i, j in groupby(tempSimulation)]
                    c = 0
                    for gg in range(len(self.groups)):
                        getInds = rm.sample(range(1, 100000), len(self.groups[gg]))
                        tempPC1s = self.gevCopulaSims[int(self.groups[gg][0])-1][getInds[0], 0]
                        tempPC2s = self.gevCopulaSims[int(self.groups[gg][0])-1][getInds[0], 1]
                        tempPC1[c:c + len(self.groups[gg])] = tempPC1s
                        tempPC2[c:c + len(self.groups[gg])] = tempPC2s
                        c = c + len(self.groups[gg])
                    self.rmm1Sims.append(tempPC1)
                    self.rmm2Sims.append(tempPC2)
                self.mjoHistoricalSimRmm1 = self.rmm1Sims
                self.mjoHistoricalSimRmm2 = self.rmm2Sims

                outputSamples['historicalDatesSim'] = self.historicalDatesSim
                outputSamples['mjoHistoricalSimRmm1'] = self.mjoHistoricalSimRmm1
                outputSamples['mjoHistoricalSimRmm2'] = self.mjoHistoricalSimRmm2
                outputSamples['historicalBmusSim'] = self.historicalBmusSim

            if futureSimNum > 0:
                futureSims = []
                for simIndex in range(futureSimNum):
                    # ALR FUTURE model simulations
                    sim_years = 100
                    # start simulation at PCs available data
                    d1 = datetime(2023, 6, 1)  # x2d(xds_cov_fit.time[0])
                    d2 = datetime(2123, 6, 1)  # datetime(d1.year+sim_years, d1.month, d1.day)
                    self.future_dates_sim = [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]

                    d1 = datetime(2023, 6, 1)
                    dt = datetime(2023, 6, 1)
                    end = datetime(2123, 6, 1)
                    step = relativedelta(years=1)
                    simAnnualTime = []
                    while dt < end:
                        simAnnualTime.append(dt)
                        dt += step

                    d1 = datetime(2023, 6, 1)
                    dt = datetime(2023, 6, 1)
                    end = datetime(2123, 6, 2)
                    # step = datetime.timedelta(months=1)
                    step = relativedelta(days=1)
                    simDailyTime = []
                    while dt < end:
                        simDailyTime.append(dt)
                        dt += step
                    simDailyDatesMatrix = np.array([[r.year, r.month, r.day] for r in simDailyTime])

                    trainingDates = [datetime(r[0], r[1], r[2]) for r in simDailyDatesMatrix]
                    dailyAWTsim = np.ones((len(trainingDates),))
                    dailyPC1sim = np.ones((len(trainingDates),))
                    dailyPC2sim = np.ones((len(trainingDates),))
                    dailyPC3sim = np.ones((len(trainingDates),))

                    awtBMUsim = self.awtBmusSim[simIndex][0:100]  # [0:len(awt_bmus)]
                    awtPC1sim = self.pc1Sims[simIndex][0:100]  # [0:len(awt_bmus)]
                    awtPC2sim = self.pc2Sims[simIndex][0:100]  # [0:len(awt_bmus)]
                    awtPC3sim = self.pc3Sims[simIndex][0:100]  # [0:len(awt_bmus)]
                    dailyDatesSWTyear = np.array([r[0] for r in simDailyDatesMatrix])
                    dailyDatesSWTmonth = np.array([r[1] for r in simDailyDatesMatrix])
                    dailyDatesSWTday = np.array([r[2] for r in simDailyDatesMatrix])
                    normPC1 = awtPC1sim
                    normPC2 = awtPC2sim
                    normPC3 = awtPC3sim

                    for i in range(len(awtBMUsim)):
                        sSeason = np.where((simDailyDatesMatrix[:, 0] == simAnnualTime[i].year) & (
                                simDailyDatesMatrix[:, 1] == simAnnualTime[i].month) & (simDailyDatesMatrix[:, 2] == 1))
                        ssSeason = np.where((simDailyDatesMatrix[:, 0] == simAnnualTime[i].year + 1) & (
                                simDailyDatesMatrix[:, 1] == simAnnualTime[i].month) & (simDailyDatesMatrix[:, 2] == 1))

                        dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1] = awtBMUsim[i] * dailyAWTsim[
                                                                                       sSeason[0][0]:ssSeason[0][0] + 1]
                        dailyPC1sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC1[i] * np.ones(
                            len(dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1]), )
                        dailyPC2sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC2[i] * np.ones(
                            len(dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1]), )
                        dailyPC3sim[sSeason[0][0]:ssSeason[0][0] + 1] = normPC3[i] * np.ones(
                            len(dailyAWTsim[sSeason[0][0]:ssSeason[0][0] + 1]), )


                    # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
                    self.xds_PCs_sim = xr.Dataset(
                        {
                            'PC1': (('time',), dailyPC1sim),
                            'PC2': (('time',), dailyPC2sim),
                            'PC3': (('time',), dailyPC3sim),
                        },
                        coords={'time': [datetime(r[0], r[1], r[2]) for r in simDailyDatesMatrix]}
                    )
                    # reindex annual data to daily data
                    self.xds_PCs_sim = xr_daily(self.xds_PCs_sim)


                    d_covars_sim = xcd_daily([self.xds_PCs_sim])

                    # PCs covar
                    cov_PCs = self.xds_PCs_sim.sel(time=slice(d_covars_sim[0], d_covars_sim[-1]))
                    cov_1 = cov_PCs.PC1.values.reshape(-1, 1)
                    cov_2 = cov_PCs.PC2.values.reshape(-1, 1)
                    cov_3 = cov_PCs.PC3.values.reshape(-1, 1)

                    # MJO covars
                    # cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0], d_covars_fit[-1]))
                    # cov_4 = cov_MJO.rmm1.values.reshape(-1, 1)
                    # cov_5 = cov_MJO.rmm2.values.reshape(-1, 1)

                    # join covars and norm.
                    cov_T = np.hstack((cov_1, cov_2, cov_3))

                    # generate xarray.Dataset
                    cov_names = ['PC1', 'PC2', 'PC3']
                    self.xds_cov_sim = xr.Dataset(
                        {
                            'cov_values': (('time', 'cov_names'), cov_T),
                        },
                        coords={
                            'time': d_covars_sim,
                            'cov_names': cov_names,
                        }
                    )


                    #  launch simulation
                    xds_ALRfuture = ALRW.Simulate(
                        1, self.future_dates_sim, self.xds_cov_sim)

                    self.futureDatesSim = self.future_dates_sim

                    futureSims.append(xds_ALRfuture.evbmus_sims.values)


                # Save results for matlab plot
                self.futureBmusSim = futureSims
                # evbmus_probcum = xds_ALR.evbmus_probcum.values
                # convert synthetic markovs to PC values
                # Fill in the Markov chain bmus with RMM vales
                rmm1Sims = list()
                rmm2Sims = list()
                for kk in range(len(self.futureBmusSim)):
                    tempSimulation = self.futureBmusSim[kk]
                    tempPC1 = np.nan * np.ones((np.shape(tempSimulation)))
                    tempPC2 = np.nan * np.ones((np.shape(tempSimulation)))

                    groups = [list(j) for i, j in groupby(tempSimulation)]
                    c = 0
                    for gg in range(len(groups)):
                        getInds = rm.sample(range(1, 100000), len(groups[gg]))
                        tempPC1s = self.gevCopulaSims[int(groups[gg][0]-1)][getInds[0], 0]
                        tempPC2s = self.gevCopulaSims[int(groups[gg][0]-1)][getInds[0], 1]
                        tempPC1[c:c + len(groups[gg])] = tempPC1s
                        tempPC2[c:c + len(groups[gg])] = tempPC2s
                        c = c + len(groups[gg])
                    rmm1Sims.append(tempPC1)
                    rmm2Sims.append(tempPC2)
                self.mjoFutureSimRmm1 = rmm1Sims
                self.mjoFutureSimRmm2 = rmm2Sims


                outputSamples['mjoFutureSimRmm1'] = self.mjoFutureSimRmm1
                outputSamples['mjoFutureSimRmm2'] = self.mjoFutureSimRmm2
                outputSamples['futureBmusSim'] = self.futureBmusSim
                outputSamples['futureDatesSim'] = self.futureDatesSim


            with open(samplesPickle, 'wb') as f:
                pickle.dump(outputSamples, f)