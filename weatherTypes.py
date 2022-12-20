
import numpy as np
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset



class weatherTypes():
    '''
    Class containing camera data and functions'''

    def __init__(self, **kwargs):

        self.lonLeft = kwargs.get('lonLeft', 270)
        self.lonRight = kwargs.get('lonRight',350)
        self.latBot = kwargs.get('latBot', 10)
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

    def extractCFSR(self,printToScreen=False):
        '''
        This function is utilized for opening *.raw Argus files prior to a debayering task,
        and adds the

        Must call this function before calling any debayering functions
        '''
        # with open(self.rawPath, "rb") as my_file:
        #     self.fh = my_file
        #     cameraIO.readHeaderLines(self)
        #     cameraIO.readAIIIrawFullFrameWithTimeStamp(self)

        year = self.startTime[0]
        month = self.startTime[1]
        year2 = self.endTime[0]
        month2 = self.endTime[1]
        print('Starting extract at {}-{}'.format(year, month))

        # estelaMat = '/media/dylananderson/Elements1/ESTELA/out/NagsHead2/NagsHead2_obj.mat'

        filePaths = np.sort(os.listdir(self.slpPath))

        initFile = self.slpPath + 'prmsl.cdas1.201104.grb2.nc'

        dt = datetime(year, month, 1)
        if month2 == 12:
            end = datetime(year2 + 1, 1, 1)
        else:
            end = datetime(year2, month2 + 1, 1)
        # step = datetime.timedelta(months=1)
        step = relativedelta(months=1)
        extractTime = []
        while dt < end:
            extractTime.append(dt)
            dt += step

        data = Dataset(initFile)
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]

        pos_lon1 = np.where((lon == self.lonLeft - 2))
        pos_lon2 = np.where((lon == self.lonRight + 2))
        pos_lat2 = np.where((lat == self.latTop + 2))
        pos_lat1 = np.where((lat == self.latBot - 2))

        latitud = lat[(pos_lat2[0][0]):(pos_lat1[0][0] + 1)]
        if self.lonLeft > self.lonRight:
            longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
        else:
            longitud = lon[pos_lon1[0][0]:(pos_lon2[0][0] + 1)]
        [x, y] = np.meshgrid(longitud, latitud)

        counter = 0
        # Now need to loop through the number of files we're extracting data from
        for tt in extractTime:

            if self.slpMemory == False:
                yearExtract = tt.year
                monthExtract = tt.month

                if (yearExtract == 2011 and monthExtract >= 4) or yearExtract > 2012:
                    file = self.slpPath + 'prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                           monthExtract)
                    # file = '/users/dylananderson/documents/data/prmsl/prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                    #                                                                                        monthExtract)
                else:
                    file = self.slpPath + 'prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                          monthExtract)
                    # file = '/users/dylananderson/documents/data/prmsl/prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                    #                                                                                       monthExtract)
                data = Dataset(file)
                time = data.variables['valid_date_time']
                if printToScreen == True:

                    print('{}-{}'.format(yearExtract, monthExtract))

                # extract times and turn them into datetimes
                # years = np.empty((len(time),))
                year = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[0:4]) for i in time]
                month = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[4:6]) for i in time]
                day = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[6:8]) for i in time]
                hour = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[8:]) for i in time]
                d_vec = np.vstack((year, month, day, hour)).T
                dates = [datetime(d[0], d[1], d[2], d[3], 0, 0) for d in d_vec]

                # Extracting SLP fields, need to account for wrap around international date line
                if self.lonLeft > self.lonRight:
                    # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                    slp1 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                    slp2 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                           0:(pos_lon2[0][0] + 1)]
                    slp = np.concatenate((slp1, slp2), axis=2)
                else:
                    slp = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                          (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]

                # are we averaging to a shorter time window?
                m, n, p = np.shape(slp)
                slp_ = np.zeros((n * p, m))
                grd_ = np.zeros((n * p, m))
                for mmm in range(m):
                    slp_[:, mmm] = slp[mmm, :, :].flatten()
                    vgrad = np.gradient(slp[mmm, :, :])
                    grd_[:, mmm] = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2).flatten()
                # slp_ = slp.reshape(n*m,p)

                if self.avgTime == 0:
                    if printToScreen == True:
                        print('returning hourly values')
                    elif counter == 0:
                        print('returning hourly values')

                    slpAvg = slp_
                    grdAvg = grd_
                    datesAvg = dates
                else:
                    numWindows = int(len(time) / self.avgTime)
                    if printToScreen == True:
                        print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))
                    elif counter == 0:
                        print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))

                    c = 0
                    datesAvg = list()
                    slpAvg = np.empty((n * p, numWindows))
                    grdAvg = np.empty((n * p, numWindows))

                    for t in range(numWindows):
                        slpAvg[:, t] = np.nanmean(slp_[:, c:c + self.avgTime], axis=1)
                        grdAvg[:, t] = np.nanmean(grd_[:, c:c + self.avgTime], axis=1)
                        datesAvg.append(dates[c])
                        c = c + self.avgTime

                # are we reducing the resolution of the grid?
                if self.resolution == 0.5:
                    if printToScreen == True:
                        print('keeping full 0.5 degree resolution')
                    elif counter == 0:
                        print('keeping full 0.5 degree resolution')

                    slpDownscaled = slpAvg
                    grdDownscaled = grdAvg
                    xDownscaled = x.flatten()
                    yDownscaled = y.flatten()
                    x2 = x
                    y2 = y
                else:
                    xFlat = x.flatten()
                    yFlat = y.flatten()
                    xRem = np.fmod(xFlat, self.resolution)
                    yRem = np.fmod(yFlat, self.resolution)
                    cc = 0
                    for pp in range(len(xFlat)):
                        if xRem[pp] == 0:
                            if yRem[pp] == 0:
                                if cc == 0:
                                    ind2deg = int(pp)
                                    cc = cc + 1
                                else:
                                    ind2deg = np.hstack((ind2deg, int(pp)))
                                    cc = cc + 1
                    slpDownscaled = slpAvg[ind2deg, :]
                    grdDownscaled = grdAvg[ind2deg, :]
                    xDownscaled = xFlat[ind2deg]
                    yDownscaled = yFlat[ind2deg]
                    if printToScreen == True:
                        print('Downscaling to {} degree resolution'.format(self.resolution))
                        print('{} points rather than {} (full)'.format(len(xDownscaled), len(xFlat)))
                    elif counter == 0:
                        print('Downscaling to {} degree resolution'.format(self.resolution))
                        print('{} points rather than {} (full)'.format(len(xDownscaled), len(xFlat)))
                    reshapeIndX = np.where((np.diff(xDownscaled) > 4) | (np.diff(xDownscaled) < -4))
                    reshapeIndY = np.where((np.diff(yDownscaled) < 0))
                    x2 = xDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1)).filled()
                    y2 = yDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0][0] + 1)).filled()

                My, Mx = np.shape(y2)
                slpMem = slpDownscaled
                grdMem = grdDownscaled
                datesMem = datesAvg


            else:

                for mm in range(2):
                    if mm == 0:
                        monthExtract = tt.month - 1
                        if monthExtract == 0:
                            monthExtract = 12
                            yearExtract = tt.year - 1
                        else:
                            yearExtract = tt.year
                    else:
                        yearExtract = tt.year
                        monthExtract = tt.month

                    if (yearExtract == 2011 and monthExtract >= 4) or yearExtract > 2012:
                        file = self.slpPath + 'prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                                    monthExtract)
                        # file = '/users/dylananderson/documents/data/prmsl/prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,
                        #                                                                                        monthExtract)
                    else:
                        file = self.slpPath + 'prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                                                                                   monthExtract)
                        # file = '/users/dylananderson/documents/data/prmsl/prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,
                        #                                                                                       monthExtract)

                    data = Dataset(file)
                    time = data.variables['valid_date_time']
                    print('{}-{}'.format(yearExtract, monthExtract))

                    # extract times and turn them into datetimes
                    # years = np.empty((len(time),))
                    year = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[0:4]) for i in time]
                    month = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[4:6]) for i in time]
                    day = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[6:8]) for i in time]
                    hour = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[8:]) for i in time]
                    d_vec = np.vstack((year, month, day, hour)).T
                    dates = [datetime(d[0], d[1], d[2], d[3], 0, 0) for d in d_vec]

                    # Extracting SLP fields, need to account for wrap around international date line
                    if self.lonLeft > self.lonRight:
                        # longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
                        slp1 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1), (pos_lon1[0][0]):]
                        slp2 = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                               0:(pos_lon2[0][0] + 1)]
                        slp = np.concatenate((slp1, slp2), axis=2)
                    else:
                        slp = data.variables['PRMSL_L101'][:, (pos_lat2[0][0]):(pos_lat1[0][0] + 1),
                              (pos_lon1[0][0]):(pos_lon2[0][0] + 1)]

                # are we averaging to a shorter time window?
                m, n, p = np.shape(slp)
                slp_ = np.zeros((n * p, m))
                grd_ = np.zeros((n * p, m))
                for mmm in range(m):
                    slp_[:, mmm] = slp[mmm, :, :].flatten()
                    vgrad = np.gradient(slp[mmm, :, :])
                    grd_[:, mmm] = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2).flatten()
                # slp_ = slp.reshape(n*m,p)

                if self.avgTime == 0:
                    if printToScreen == True:
                        print('returning hourly values')
                    slpAvg = slp_
                    grdAvg = grd_
                    datesAvg = dates
                else:
                    numWindows = int(len(time) / self.avgTime)
                    if printToScreen == True:
                        print('averaging every {} hours to {} timesteps'.format(self.avgTime, numWindows))
                    c = 0
                    datesAvg = list()
                    slpAvg = np.empty((n * p, numWindows))
                    grdAvg = np.empty((n * p, numWindows))
                    for t in range(numWindows):
                        slpAvg[:, t] = np.nanmean(slp_[:, c:c + self.avgTime], axis=1)
                        grdAvg[:, t] = np.nanmean(grd_[:, c:c + self.avgTime], axis=1)
                        datesAvg.append(dates[c])
                        c = c + self.avgTime

                # are we reducing the resolution of the grid?
                if self.resolution == 0.5:
                    if printToScreen == True:
                        print('keeping full 0.5 degree resolution')
                    slpDownscaled = slpAvg
                    grdDownscaled = grdAvg
                    xDownscaled = x.flatten()
                    yDownscaled = y.flatten()
                    x2 = x
                    y2 = y
                else:
                    xFlat = x.flatten()
                    yFlat = y.flatten()
                    xRem = np.fmod(xFlat, self.resolution)
                    yRem = np.fmod(yFlat, self.resolution)
                    cc = 0
                    for pp in range(len(xFlat)):
                        if xRem[pp] == 0:
                            if yRem[pp] == 0:
                                if cc == 0:
                                    ind2deg = int(pp)
                                    cc = cc + 1
                                else:
                                    ind2deg = np.hstack((ind2deg, int(pp)))
                                    cc = cc + 1
                    slpDownscaled = slpAvg[ind2deg, :]
                    grdDownscaled = grdAvg[ind2deg, :]
                    xDownscaled = xFlat[ind2deg]
                    yDownscaled = yFlat[ind2deg]
                    if printToScreen == True:
                        print('Downscaling to {} degree resolution'.format(self.resolution))
                        print('{} points rather than {} (full)'.format(len(xDownscaled), len(xFlat)))
                    reshapeIndX = np.where((np.diff(xDownscaled) > 4) | (np.diff(xDownscaled) < -4))
                    reshapeIndY = np.where((np.diff(yDownscaled) < 0))
                    x2 = xDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0] + 1))
                    y2 = yDownscaled.reshape(int(len(reshapeIndY[0]) + 1), int(reshapeIndX[0] + 1))
                My, Mx = np.shape(y2)
                slpMem = slpDownscaled
                datesMem = datesAvg
                grdMem = grdDownscaled

                # dim, Ntime = np.shape(slpDownscaled)
                # GrdSlp = np.zeros((np.shape(slpDownscaled)))
                # for ttt in range(Ntime):
                #     p = slpDownscaled[:,ttt].reshape(My,Mx)
                #     dp = np.zeros((np.shape(p)))
                #     ii = np.arange(1,Mx-1)
                #     jj = np.arange(1,My-1)
                #     for iii in ii:
                #         for jjj in jj:
                #             phi = np.pi*np.abs(y2[jjj,iii])/180
                #             dpx1 = (p[jjj,iii] - p[jjj,iii-1])/np.cos(phi)
                #             dpx2 = (p[jjj,iii+1] - p[jjj,iii])/np.cos(phi)
                #             dpy1 = p[jjj,iii] - p[jjj-1,iii]
                #             dpy2 = p[jjj+1,iii] - p[jjj,iii]
                #             dp[jjj,iii] = (dpx1**2 + dpx2**2 )/2+(dpy1**2 + dpy2**2)/2
                #     GrdSlp[:,ttt] = dp.flatten()

                # if slpMemory == True:
                #
                #     with h5py.File(estelaMat, 'r') as f:
                #         # for k in f.keys():
                #         #     print(k)
                #         Xraw = f['full/Xraw'][:]
                #         Y = f['full/Y'][:]
                #         DJF = f['C/traveldays_interp/DJF'][:]
                #         MAM = f['C/traveldays_interp/MAM'][:]
                #         JJA = f['C/traveldays_interp/JJA'][:]
                #         SON = f['C/traveldays_interp/SON'][:]
                #
                #
                #
                #     X_estela = Xraw
                #     tempX = np.where(X_estela < 120)
                #     X_estela[tempX] = X_estela[tempX]+(X_estela[tempX]*0+360)
                #     estelaX = np.hstack((X_estela[:,600:],X_estela[:,0:600]))
                #
                #     Y_estela = Y
                #     estelaY = np.hstack((Y_estela[:,600:],Y_estela[:,0:600]))
                #
                #     #dateTemp = days
                # if mm == 0:
                #     slpMem = slpDownscaled
                #     datesMem = datesAvg
                # else:
                #     slpMem = np.hstack((slpMem, slpDownscaled))
                #     datesMem = np.append(datesMem, datesAvg)
            # slpMem = slpDownscaled
            # datesMem = datesAvg


            if counter == 0:
                SLPS = slpMem
                GRDS = grdMem
                DATES = datesMem
            else:
                SLPS = np.hstack((SLPS, slpMem))
                GRDS = np.hstack((GRDS, grdMem))
                DATES = np.append(DATES, datesMem)

            counter = counter + 1

        # vgrad = np.gradient(SLPS)
        # magGrad = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2)


        print('Extracted until {}-{}'.format(year2, month2))

        from global_land_mask import globe
        wrapLons = np.where((x2 > 180))
        x2[wrapLons] = x2[wrapLons] - 360
        xFlat = x2.flatten()
        yFlat = y2.flatten()

        isOnLandGrid = globe.is_land(y2, x2)
        isOnLandFlat = globe.is_land(yFlat, xFlat)

        x2[wrapLons] = x2[wrapLons] + 360
        xFlat = x2.flatten()


        self.xGrid = x2
        self.yGrid = y2
        self.xFlat = xFlat
        self.yFlat = yFlat
        self.isOnLandGrid = isOnLandGrid
        self.isOnLandFlat = isOnLandFlat
        self.SLPS = SLPS
        self.GRDS = GRDS
        self.DATES = DATES
        self.Mx = Mx
        self.My = My



    def pcaOfSlps(self):

        from sklearn.decomposition import PCA

        trimSlps = self.SLPS[~self.isOnLandFlat,:]
        trimGrds = self.GRDS[~self.isOnLandFlat,:]


        SlpGrd = np.hstack((trimSlps.T, trimGrds.T))
        SlpGrdMean = np.mean(SlpGrd, axis=0)
        SlpGrdStd = np.std(SlpGrd, axis=0)
        SlpGrdNorm = (SlpGrd[:, :] - SlpGrdMean) / SlpGrdStd
        SlpGrdNorm[np.isnan(SlpGrdNorm)] = 0

        # principal components analysis
        ipca = PCA(n_components=min(SlpGrdNorm.shape[0], SlpGrdNorm.shape[1]))
        PCs = ipca.fit_transform(SlpGrdNorm)
        EOFs = ipca.components_
        variance = ipca.explained_variance_
        nPercent = variance / np.sum(variance)
        APEV = np.cumsum(variance) / np.sum(variance) * 100.0
        nterm = np.where(APEV <= 0.95 * 100)[0][-1]

        self.SlpGrdMean = SlpGrdMean
        self.SlpGrdStd = SlpGrdStd
        self.SlpGrdNorm = SlpGrdNorm
        self.SlpGrd = SlpGrd
        self.PCs = PCs
        self.EOFs = EOFs
        self.variance = variance
        self.nPercent = nPercent
        self.APEV = APEV
        self.nterm = nterm



    def wtClusters(self,numClusters=49):
        from functions import sort_cluster_gen_corr_end
        from sklearn.cluster import KMeans
        self.numClusters = numClusters
        PCsub = self.PCs[:, :self.nterm + 1]
        EOFsub = self.EOFs[:self.nterm + 1, :]
        kma = KMeans(n_clusters=numClusters, n_init=2000).fit(PCsub)
        # groupsize
        _, group_size = np.unique(kma.labels_, return_counts=True)
        # groups
        d_groups = {}
        for k in range(numClusters):
            d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
        self.groupSize = group_size
        self.dGroups = d_groups
        # centroids
        centroids = np.dot(kma.cluster_centers_, EOFsub)
        # km, x and var_centers
        km = np.multiply(
            centroids,
            np.tile(self.SlpGrdStd, (numClusters, 1))
        ) + np.tile(self.SlpGrdMean, (numClusters, 1))
        # sort kmeans
        kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, numClusters)
        self.kmaOrder = kma_order
        bmus_corrected = np.zeros((len(kma.labels_),), ) * np.nan
        for i in range(numClusters):
            posc = np.where(kma.labels_ == kma_order[i])
            bmus_corrected[posc] = i
        self.bmus_corrected = bmus_corrected
        # reorder centroids
        self.sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
        self.sorted_centroids = centroids[kma_order, :]

        repmatStd = np.tile(self.SlpGrdStd, (numClusters, 1))
        repmatMean = np.tile(self.SlpGrdMean, (numClusters, 1))
        self.Km_ = np.multiply(self.sorted_centroids, repmatStd) + repmatMean





