


def plotOceanConditions(struct):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    plt.figure()

    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
    ax1.plot(struct.timeWave, struct.Hs, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax1.set_ylabel('Hs (m)')

    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=1, colspan=1)
    ax2.plot(struct.timeWave, struct.Tp, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax2.set_ylabel('Tp (s)')

    ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1)
    ax3.plot(struct.timeWave, struct.Dm, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax3.set_ylabel('Dm')

    ax4 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)
    ax4.plot(struct.timeWL, struct.waterLevel, color=[0.65, 0.65, 0.65], linewidth=0.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('wl (m)')


def plotWTs(struct,withTCs=True):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    if withTCs == True:
        dwtcolors = cm.rainbow(np.linspace(0, 1, struct.numClusters+struct.num_clustersTC))

        # plotting the EOF patterns
        fig2 = plt.figure(figsize=(10, 10))
        # gs1 = gridspec.GridSpec(int(np.ceil(np.sqrt(struct.numClustersETC))), int(np.ceil(np.sqrt(struct.numClustersETC))))
        gs1 = gridspec.GridSpec(int(10), int(7))

        gs1.update(wspace=0.00, hspace=0.00)  # set the spacing between axes.
        c1 = 0
        c2 = 0
        counter = 0
        plotIndx = 0
        plotIndy = 0
        for hh in range(struct.numClusters):
            # p1 = plt.subplot2grid((6,6),(c1,c2))
            ax = plt.subplot(gs1[hh])
            num = struct.kmaOrderETC[hh]

            spatialField = struct.Km_ETC[(hh), 0:len(struct.xFlat[~struct.isOnLandFlat])] / 100 - struct.SlpGrdMean[0:len(
                struct.xFlat[~struct.isOnLandFlat])] / 100
            # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))
            X_in = struct.xFlat[~struct.isOnLandFlat]
            Y_in = struct.yFlat[~struct.isOnLandFlat]
            sea_nodes = []
            for qq in range(len(X_in)):
                sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

            rectField = np.ones((np.shape(struct.xGrid))) * np.nan
            for tt in range(len(sea_nodes)):
                rectField[sea_nodes[tt]] = spatialField[tt]

            clevels = np.arange(-32, 32, 1)
            # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
            #             resolution='c')
            m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=45, llcrnrlon=255, urcrnrlon=345, lat_ts=20,
                        resolution='c')
            m.fillcontinents(color=dwtcolors[hh])
            cx, cy = m(struct.xGrid, struct.yGrid)
            m.drawcoastlines()
            CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)  # , shading='gouraud')
            # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
            tx, ty = m(320, 10)
            ax.text(tx, ty, '{}'.format(struct.groupSizeETC[num]))

            c2 += 1
            if c2 == int(np.ceil(np.sqrt(struct.numClusters)) - 1):
                c1 += 1
                c2 = 0

            if plotIndx <= np.ceil(np.sqrt(struct.numClusters)):
                ax.xaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
            if plotIndy > 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks([])
            counter = counter + 1
            if plotIndy <= np.ceil(np.sqrt(struct.numClusters)):
                plotIndy = plotIndy + 1
            else:
                plotIndy = 0
                plotIndx = plotIndx + 1

        for hh in range(struct.num_clustersTC):
            # p1 = plt.subplot2grid((6,6),(c1,c2))
            ax = plt.subplot(gs1[hh+49])
            num = hh#struct.kmaOrderTC[hh]

            spatialField = struct.Km_TC[(hh), 0:len(struct.xFlat[~struct.isOnLandFlat])] / 100 - struct.SlpGrdMean[0:len(
                struct.xFlat[~struct.isOnLandFlat])] / 100
            # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))
            X_in = struct.xFlat[~struct.isOnLandFlat]
            Y_in = struct.yFlat[~struct.isOnLandFlat]
            sea_nodes = []
            for qq in range(len(X_in)):
                sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

            rectField = np.ones((np.shape(struct.xGrid))) * np.nan
            for tt in range(len(sea_nodes)):
                rectField[sea_nodes[tt]] = spatialField[tt]

            clevels = np.arange(-32, 32, 1)
            # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
            #             resolution='c')
            m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=45, llcrnrlon=255, urcrnrlon=345, lat_ts=20,
                        resolution='c')
            m.fillcontinents(color=dwtcolors[hh+49])
            cx, cy = m(struct.xGrid, struct.yGrid)
            m.drawcoastlines()
            CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)  # , shading='gouraud')
            # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
            tx, ty = m(320, 10)
            ax.text(tx, ty, '{}'.format(struct.groupSizeTC[num]))

            c2 += 1
            if c2 == int(np.ceil(np.sqrt(struct.numClusters)) - 1):
                c1 += 1
                c2 = 0

            if plotIndx <= np.ceil(np.sqrt(struct.numClusters)):
                ax.xaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
            if plotIndy > 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks([])
            counter = counter + 1
            if plotIndy <= np.ceil(np.sqrt(struct.numClusters)):
                plotIndy = plotIndy + 1
            else:
                plotIndy = 0
                plotIndx = plotIndx + 1


    else:
        dwtcolors = cm.rainbow(np.linspace(0, 1, struct.numClusters))

        # plotting the EOF patterns
        fig2 = plt.figure(figsize=(10, 10))
        gs1 = gridspec.GridSpec(int(np.ceil(np.sqrt(struct.numClusters))), int(np.ceil(np.sqrt(struct.numClusters))))
        gs1.update(wspace=0.00, hspace=0.00)  # set the spacing between axes.
        c1 = 0
        c2 = 0
        counter = 0
        plotIndx = 0
        plotIndy = 0
        for hh in range(struct.numClusters):
            # p1 = plt.subplot2grid((6,6),(c1,c2))
            ax = plt.subplot(gs1[hh])
            num = struct.kmaOrderETC[hh]

            spatialField = struct.Km_ETC[(hh), 0:len(struct.xFlat[~struct.isOnLandFlat])] / 100 - struct.SlpGrdMean[0:len(
                struct.xFlat[~struct.isOnLandFlat])] / 100
            # spatialField = np.multiply(EOFs[hh, len(self.xFlat[~self.isOnLandFlat]):], np.sqrt(variance[hh]))
            X_in = struct.xFlat[~struct.isOnLandFlat]
            Y_in = struct.yFlat[~struct.isOnLandFlat]
            sea_nodes = []
            for qq in range(len(X_in)):
                sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

            rectField = np.ones((np.shape(struct.xGrid))) * np.nan
            for tt in range(len(sea_nodes)):
                rectField[sea_nodes[tt]] = spatialField[tt]

            clevels = np.arange(-32, 32, 1)
            # m = Basemap(projection='merc', llcrnrlat=2, urcrnrlat=52, llcrnrlon=270, urcrnrlon=360, lat_ts=25,
            #             resolution='c')
            m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=45, llcrnrlon=255, urcrnrlon=345, lat_ts=20,
                        resolution='c')
            m.fillcontinents(color=dwtcolors[hh])
            cx, cy = m(struct.xGrid, struct.yGrid)
            m.drawcoastlines()
            CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r)  # , shading='gouraud')
            # p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
            tx, ty = m(320, 10)
            ax.text(tx, ty, '{}'.format(struct.groupSizeETC[num]))

            c2 += 1
            if c2 == int(np.ceil(np.sqrt(struct.numClusters)) - 1):
                c1 += 1
                c2 = 0

            if plotIndx <= np.ceil(np.sqrt(struct.numClusters)):
                ax.xaxis.set_ticks([])
                ax.xaxis.set_ticklabels([])
            if plotIndy > 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks([])
            counter = counter + 1
            if plotIndy <= np.ceil(np.sqrt(struct.numClusters)):
                plotIndy = plotIndy + 1
            else:
                plotIndy = 0
                plotIndx = plotIndx + 1


def plotSlpExample(struct, plotTime=0):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    fig = plt.figure(figsize=(10, 6))

    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    clevels = np.arange(940, 1080, 1)

    spatialField = struct.SLPS[:, plotTime] / 100  # SLPS[:, i] / 100  # - np.nanmean(SLP, axis=1) / 100

    rectField = spatialField.reshape(struct.My, struct.Mx)
    # rectField[~is_on_land] = rectField[~is_on_land]*np.nan
    rectFieldMasked = np.where(~struct.isOnLandGrid, rectField, 0)
    m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10,
                resolution='c')
    m.fillcontinents(color=[0.5, 0.5, 0.5])
    cx, cy = m(struct.xGrid, struct.yGrid)

    m.drawcoastlines()
    # m.bluemarble()
    # CS = m.contourf(cx, cy, rectField.T, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r, shading='gouraud')
    CS = m.contourf(cx.T, cy.T, rectFieldMasked.T, clevels, vmin=975, vmax=1045, cmap=cm.RdBu_r)  # , shading='gouraud')

    tx, ty = m(320, -0)
    parallels = np.arange(0, 360, 10)
    m.drawparallels(parallels, labels=[True, True, True, False], textcolor='black')
    # ax.text(tx, ty, '{}'.format((group_size[num])))
    meridians = np.arange(0, 360, 20)
    m.drawmeridians(meridians, labels=[True, True, True, True], textcolor='black')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(CS, cax=cbar_ax)
    cbar.set_label('SLP (mbar)')



def plotEOFs(struct):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    # plotting the EOF patterns for SLPs
    plt.figure()
    c1 = 0
    c2 = 0
    for hh in range(9):
        p1 = plt.subplot2grid((3, 3), (c1, c2))
        spatialField = np.multiply(struct.EOFs[hh, 0:len(struct.xFlat[~struct.isOnLandFlat])], np.sqrt(struct.variance[hh]))
        X_in = struct.xFlat[~struct.isOnLandFlat]
        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-2, 2, .05)
        m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
                    resolution='c')
        cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
        c2 += 1
        if c2 == 3:
            c1 += 1
            c2 = 0

    # plotting the EOF patterns for GRDs
    plt.figure()
    c1 = 0
    c2 = 0
    for hh in range(9):
        p1 = plt.subplot2grid((3, 3), (c1, c2))
        spatialField = np.multiply(struct.EOFs[hh, len(struct.xFlat[~struct.isOnLandFlat]):], np.sqrt(struct.variance[hh]))
        X_in = struct.xFlat[~struct.isOnLandFlat]
        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-2, 2, .05)
        m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
                        resolution='c')
        cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
        c2 += 1
        if c2 == 3:
            c1 += 1
            c2 = 0



def plotEOFsAndPCs(struct):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.basemap import Basemap
    # plotting the EOF patterns for SLPs
    plt.figure()
    c1 = 0
    c2 = 0
    for hh in range(9):
        p1 = plt.subplot2grid((9, 3), (c1, c2),rowspan=2,colspan=1)
        spatialField = np.multiply(struct.EOFs[hh, 0:len(struct.xFlat[~struct.isOnLandFlat])], np.sqrt(struct.variance[hh]))
        X_in = struct.xFlat[~struct.isOnLandFlat]
        Y_in = struct.yFlat[~struct.isOnLandFlat]
        sea_nodes = []
        for qq in range(len(X_in)):
            sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))

        rectField = np.ones((np.shape(struct.xGrid))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-2, 2, .05)
        # m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
        #             resolution='c')
        m = Basemap(projection='merc', llcrnrlat=5, urcrnrlat=45, llcrnrlon=255, urcrnrlon=360, lat_ts=20,
                    resolution='c')
        cx, cy = m(struct.xGrid, struct.yGrid)
        m.drawcoastlines()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
        p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
        p1 = plt.subplot2grid((9, 3), (c1+2, c2),rowspan=1,colspan=1)
        p1.plot(struct.DATES, struct.PCs[:, hh])
        c2 += 1
        if c2 == 3:
            c1 += 3
            c2 = 0


    #
    # c1 = 3
    # c2 = 0
    # for hh in range(9):
    #     p1 = plt.subplot2grid((9, 3), (c1, c2),rowspan=1,colspan=1)
    #     p1.plot(struct.DATES,struct.PCs[:,hh])
    #     # spatialField = np.multiply(struct.EOFs[hh, 0:len(struct.xFlat[~struct.isOnLandFlat])], np.sqrt(struct.variance[hh]))
    #     # X_in = struct.xFlat[~struct.isOnLandFlat]
    #     # Y_in = struct.yFlat[~struct.isOnLandFlat]
    #     # sea_nodes = []
    #     # for qq in range(len(X_in)):
    #     #     sea_nodes.append(np.where((struct.xGrid == X_in[qq]) & (struct.yGrid == Y_in[qq])))
    #     #
    #     # rectField = np.ones((np.shape(struct.xGrid))) * np.nan
    #     # for tt in range(len(sea_nodes)):
    #     #     rectField[sea_nodes[tt]] = spatialField[tt]
    #     #
    #     # clevels = np.arange(-2, 2, .05)
    #     # m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10,
    #     #             resolution='c')
    #     # cx, cy = m(struct.xGrid, struct.yGrid)
    #     # m.drawcoastlines()
    #     # CS = m.contourf(cx, cy, rectField, clevels, vmin=-1.2, vmax=1.2, cmap=cm.RdBu_r)  # , shading='gouraud')
    #     # # p1.set_title('EOF {} = {}%'.format(hh + 1, np.round(struct.nPercent[hh] * 10000) / 100))
    #     c2 += 1
    #     if c2 == 3:
    #         c1 += 3
    #         c2 = 0



def plotSeasonal(struct):
    from datetime import datetime, timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    def GenOneYearDaily(yy=1981, month_ini=1):
        'returns one generic year in a list of datetimes. Daily resolution'

        dp1 = datetime(yy, month_ini, 2)
        dp2 = dp1 + timedelta(days=364)

        return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]

    def GenOneSeasonDaily(yy=1981, month_ini=1):
        'returns one generic year in a list of datetimes. Daily resolution'

        dp1 = datetime(yy, month_ini, 1)
        dp2 = dp1 + timedelta(3 * 365 / 12)

        return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]

    # tC = [datetime.utcfromtimestamp((qq - np.datetime64(0, 's')) / np.timedelta64(1, 's')) for qq in struct.DATES]
    tC = struct.DATES

    bmus_dates_months = np.array([d.month for d in tC])
    bmus_dates_days = np.array([d.day for d in tC])


    # generate perpetual year list
    list_pyear = GenOneYearDaily(month_ini=6)
    # m_plot = np.zeros((struct.numClusters, len(list_pyear))) * np.nan
    m_plot = np.zeros((int(np.max(struct.bmus_corrected)+1), len(list_pyear))) * np.nan

    numberOfSims = 1
    # sort data
    for i, dpy in enumerate(list_pyear):
       _, s = np.where(
          [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
       )
       b = struct.bmus_corrected[s]
       b = b.flatten()

       for j in range(int(np.max(struct.bmus_corrected)+1)):
          _, bb = np.where([(j == b)])  # j+1 starts at 1 bmus value!

          m_plot[j, i] = float(len(bb) / float(numberOfSims)) / len(s)

    import matplotlib.cm as cm
    # dwtcolors = cm.viridis(np.linspace(0, 1, struct.numClusters))
    # dwtcolors = cm.rainbow(np.linspace(0, 1, struct.numClusters))
    dwtcolors = cm.rainbow(np.linspace(0, 1, int(np.max(struct.bmus_corrected)+1)))


    fig = plt.figure()
    ax = plt.subplot2grid((1,1),(0,0))
    # plot stacked bars
    bottom_val = np.zeros(m_plot[1, :].shape)
    for r in range(int(np.max(struct.bmus_corrected)+1)):
       row_val = m_plot[r, :]
       ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
       # store bottom
       bottom_val += row_val

    import matplotlib.dates as mdates
    # customize  axis
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    ax.set_xlim(list_pyear[0], list_pyear[-1])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.set_ylim(0, 1)#struct.endTime[0]-struct.startTime[0])
    ax.set_ylabel('')
