import datetime
import random
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
from scipy.spatial import distance_matrix
from datetime import timedelta
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.special import ndtri  # norm inv
from scipy.stats import norm, genpareto, t
from statsmodels.distributions.empirical_distribution import ECDF

from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min


def loadWaterLevel(file):
    from netCDF4 import Dataset
    wldata = Dataset(file)
    waterLevel = wldata.variables['waterLevel'][:]
    predictedWaterLevel = wldata.variables['predictedWaterLevel'][:]
    residualWaterLevel = wldata.variables['residualWaterLevel'][:]
    timeWl = wldata.variables['time'][:]
    output = dict()
    output['waterLevel'] = waterLevel
    output['predictedWaterLevel'] = predictedWaterLevel
    output['residualWaterLevel'] = residualWaterLevel
    output['time'] = timeWl
    return output


def loadWIS(file):
    from netCDF4 import Dataset

    waves = Dataset(file)
    waveHs = waves.variables['waveHs'][:]
    waveTp = waves.variables['waveTp'][:]
    waveMeanDirection = waves.variables['waveMeanDirection'][:]
    waveTm = waves.variables['waveTm'][:]
    waveTm1 = waves.variables['waveTm1'][:]
    waveTm2 = waves.variables['waveTm2'][:]
    waveHsWindsea = waves.variables['waveHsWindsea'][:]
    waveTmWindsea = waves.variables['waveTmWindsea'][:]
    waveMeanDirectionWindsea = waves.variables['waveMeanDirectionWindsea'][:]
    waveSpreadWindsea = waves.variables['waveSpreadWindsea'][:]
    timeW = waves.variables['time'][:]
    waveTpSwell = waves.variables['waveTpSwell'][:]
    waveHsSwell = waves.variables['waveHsSwell'][:]
    waveMeanDirectionSwell = waves.variables['waveMeanDirectionSwell'][:]
    waveSpreadSwell = waves.variables['waveSpreadSwell'][:]

    output = dict()
    output['waveHs'] = waveHs
    output['waveTp'] = waveTp
    output['waveMeanDirection'] = waveMeanDirection
    output['waveTm'] = waveTm
    output['waveTm1'] = waveTm1
    output['waveTm2'] = waveTm2
    output['waveTpSwell'] = waveTpSwell
    output['waveHsSwell'] = waveHsSwell
    output['waveMeanDirectionSwell'] = waveMeanDirectionSwell
    output['waveSpreadSwell'] = waveSpreadSwell
    output['waveHsWindsea'] = waveHsWindsea
    output['waveTpWindsea'] = waveTmWindsea
    output['waveMeanDirectionWindsea'] = waveMeanDirectionWindsea
    output['waveSpreadWindsea'] = waveSpreadWindsea
    output['t'] = timeW

    return output



def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def dateDay2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetimeDate(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.date(d[0], d[1], d[2]) for d in d_vec]


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out



def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''
    # TODO: DOCUMENTAR.

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
        # TODO: RAISE ERROR
        pass

    dd = distance_matrix(centers, centers)
    qx = 0
    sc = np.random.permutation(dimdim).reshape(dimy, dimx)

    # get qx
    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i==0:
                qx += dd[sc[i-1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i-1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i-1,j+1], sc[i,j]]

            # row F
            if not j==0:
                qx += dd[sc[i,j-1], sc[i,j]]

            if not j+1==dimx:
                qx += dd[sc[i,j+1], sc[i,j]]

            # row F+1
            if not i+1==dimy:
                qx += dd[sc[i+1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i+1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i+1,j+1], sc[i,j]]

    # test permutations
    q=np.inf
    go_out = False
    for i in range(dimdim):
        if go_out:
            break

        go_out = True

        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i,j,k]))==3:

                    u = sc.flatten('F')
                    u[i] = sc.flatten('F')[j]
                    u[j] = sc.flatten('F')[k]
                    u[k] = sc.flatten('F')[i]
                    u = u.reshape(dimy, dimx, order='F')

                    f=0
                    for ix in range(dimy):
                        for jx in range(dimx):

                            # row F-1
                            if not ix==0:
                                f += dd[u[ix-1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix-1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix-1,jx+1], u[ix,jx]]

                            # row F
                            if not jx==0:
                                f += dd[u[ix,jx-1], u[ix,jx]]

                            if not jx+1==dimx:
                                f += dd[u[ix,jx+1], u[ix,jx]]

                            # row F+1
                            if not ix+1==dimy:
                                f += dd[u[ix+1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix+1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix+1,jx+1], u[ix,jx]]

                    if f<=q:
                        q = f
                        sc = u

                        if q<=qx:
                            qx=q
                            go_out=False

    return sc.flatten('F')



def fitGEVparams(var):
    '''
    Returns stationary GEV/Gumbel_L params for KMA bmus and varible series

    bmus        - KMA bmus (time series of KMA centroids)
    n_clusters  - number of KMA clusters
    var         - time series of variable to fit to GEV/Gumbel_L

    returns np.array (n_clusters x parameters). parameters = (shape, loc, scale)
    for gumbel distributions shape value will be ~0 (0.0000000001)
    '''

    param_GEV = np.empty((3,))

    # get variable at cluster position
    var_c = var
    var_c = var_c[~np.isnan(var_c)]

    # fit to Gumbel_l and get negative loglikelihood
    loc_gl, scale_gl = gumbel_l.fit(-var_c)
    theta_gl = (0.0000000001, -1*loc_gl, scale_gl)
    nLogL_gl = genextreme.nnlf(theta_gl, var_c)

    # fit to GEV and get negative loglikelihood
    c = -0.1
    shape_gev, loc_gev, scale_gev = genextreme.fit(var_c, c)
    theta_gev = (shape_gev, loc_gev, scale_gev)
    nLogL_gev = genextreme.nnlf(theta_gev, var_c)

    # store negative shape
    theta_gev_fix = (-shape_gev, loc_gev, scale_gev)

    # apply significance test if Frechet
    if shape_gev < 0:

        # TODO: cant replicate ML exact solution
        if nLogL_gl - nLogL_gev >= 1.92:
            param_GEV = list(theta_gev_fix)
        else:
            param_GEV = list(theta_gl)
    else:
        param_GEV = list(theta_gev_fix)

    return param_GEV


def gev_CDF(x):
    '''
    :param x: observations
    :return: normalized cdf
    '''
    shape, loc, scale = fitGEVparams(x)
    cdf = genextreme.cdf(x, -1 * shape, loc, scale)
    return cdf

def gev_ICDF(x,y):
    '''
    :param x: observations
    :param y: simulated probabilities
    :return: simulated values
    '''
    shape, loc, scale = fitGEVparams(x)
    ppf_VV = genextreme.ppf(y, -1 * shape, loc, scale)
    return ppf_VV


def ksdensity_CDF(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF at x position (kde.support = x)
    fint = interp1d(kde.support, kde.cdf)
    return fint(x)

def ksdensity_ICDF(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)
    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]
    return fint(p)

def GeneralizedPareto_CDF(x):
    '''
    Generalized Pareto fit
    Returns cumulative probability function at x.
    '''

    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)

    # get generalized pareto CDF
    cdf = genpareto.cdf(x, shape, scale=scale)

    return cdf

def GeneralizedPareto_ICDF(x, p):
    '''
    Generalized Pareto fit
    Returns inverse cumulative probability function at p points
    '''
    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)
    # get percent points (inverse of CDF)
    icdf = genpareto.ppf(p, shape, scale=scale)
    return icdf

def Empirical_CDF(x):
    '''
    Returns empirical cumulative probability function at x.
    '''
    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)
    return cdf

def Empirical_ICDF(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: build in functionality for a fill_value?

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)


def copulafit(u, family='gaussian'):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student
    '''

    rhohat = None  # correlation matrix
    nuhat = None  # degrees of freedom (for t student)

    if family=='gaussian':
        u[u>=1.0] = 0.999999
        inv_n = ndtri(u)
        rhohat = np.corrcoef(inv_n.T)

    elif family=='t':
        raise ValueError("Not implemented")

        # # :
        # x = np.linspace(np.min(u), np.max(u),100)
        # inv_t = np.ndarray((len(x), u.shape[1]))
        #
        # for j in range(u.shape[1]):
        #     param = t.fit(u[:,j])
        #     t_pdf = t.pdf(x,loc=param[0],scale=param[1],df=param[2])
        #     inv_t[:,j] = t_pdf
        #
        # # CORRELATION? NUHAT?
        # rhohat = np.corrcoef(inv_n.T)
        # nuhat = None

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return rhohat, nuhat

def copularnd(family, rhohat, n):
    '''
    Random vectors from a copula
    '''

    if family=='gaussian':
        mn = np.zeros(rhohat.shape[0])
        np_rmn = np.random.multivariate_normal(mn, rhohat, n)
        u = norm.cdf(np_rmn)

    elif family=='t':
        # TODO
        raise ValueError("Not implemented")

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return u


def copulaSimulation(U_data, kernels, num_sim):
    '''
    Fill statistical space using copula simulation

    U_data: 2D nump.array, each variable in a column
    kernels: list of kernels for each column at U_data (KDE | GPareto | Empirical | GEV)
    num_sim: number of simulations
    '''

    # kernel CDF dictionary
    d_kf = {
        'KDE' : (ksdensity_CDF, ksdensity_ICDF),
        'GPareto' : (GeneralizedPareto_CDF, GeneralizedPareto_ICDF),
        'ECDF' : (Empirical_CDF, Empirical_ICDF),
        'GEV': (gev_CDF, gev_ICDF),
    }


    # check kernel input
    if any([k not in d_kf.keys() for k in kernels]):
        raise ValueError(
            'wrong kernel: {0}, use: {1}'.format(
                kernels, ' | '.join(d_kf.keys())
            )
        )


    # normalize: calculate data CDF using kernels
    U_cdf = np.zeros(U_data.shape) * np.nan
    ic = 0
    for d, k in zip(U_data.T, kernels):
        cdf, _ = d_kf[k]  # get kernel cdf
        U_cdf[:, ic] = cdf(d)
        ic += 1

    # fit data CDFs to a gaussian copula
    rhohat, _ = copulafit(U_cdf, 'gaussian')

    # simulate data to fill probabilistic space
    U_cop = copularnd('gaussian', rhohat, num_sim)

    # de-normalize: calculate data ICDF
    U_sim = np.zeros(U_cop.shape) * np.nan
    ic = 0
    for d, c, k in zip(U_data.T, U_cop.T, kernels):
        _, icdf = d_kf[k]  # get kernel icdf
        U_sim[:, ic] = icdf(d, c)
        ic += 1

    return U_sim





def xds2datetime(d64):
    from datetime import datetime
    'converts xr.Dataset np.datetime64[ns] into datetime'
    # TODO: hour minutes and seconds

    return datetime(int(d64.dt.year), int(d64.dt.month), int(d64.dt.day))


def xds_reindex_daily(xds_data,  dt_lim1=None, dt_lim2=None):
    from datetime import datetime
    '''
    Reindex xarray.Dataset to daily data between optional limits
    '''

    # TODO: remove limits from inside function

    # TODO: remove this swich and use date2datenum
    if isinstance(xds_data.time.values[0], datetime):
        xds_dt1 = xds_data.time.values[0]
        xds_dt2 = xds_data.time.values[-1]
    else:
        # parse xds times to python datetime
        xds_dt1 = xds2datetime(xds_data.time[0])
        xds_dt2 = xds2datetime(xds_data.time[-1])

    # cut data at limits
    if dt_lim1:
        xds_dt1 = max(xds_dt1, dt_lim1)
    if dt_lim2:
        xds_dt2 = min(xds_dt2, dt_lim2)

    # number of days
    num_days = (xds_dt2-xds_dt1).days+1

    # reindex xarray.Dataset
    return xds_data.reindex(
        {'time': [xds_dt1 + timedelta(days=i) for i in range(num_days)]},
        method = 'pad',
    )




def xds_common_dates_daily(xds_list):
    from datetime import timedelta
    '''
    returns daily datetime array between a list of xarray.Dataset comon date
    limits
    '''
    d1, d2 = xds_limit_dates(xds_list)
    return [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]


def xds_limit_dates(xds_list):
    '''
    returns datetime common limits between a list of xarray.Dataset
    '''
    from datetime import datetime
    d1 = None
    d2 = None

    for xds_e in xds_list:

        # TODO: remove this swich and use date2datenum
        if isinstance(xds_e.time.values[0], datetime):
            xds_e_dt1 = xds_e.time.values[0]
            xds_e_dt2 = xds_e.time.values[-1]
        else:
            # parse xds times to python datetime
            xds_e_dt1 = xds2datetime(xds_e.time[0])
            xds_e_dt2 = xds2datetime(xds_e.time[-1])

        if d1 == None:
            d1 = xds_e_dt1
            d2 = xds_e_dt2

        d1 = max(xds_e_dt1, d1)
        d2 = min(xds_e_dt2, d2)

    return d1, d2



def npdt64todatetime(dt64):
    from datetime import datetime
    from datetime import timedelta
    'converts np.datetime64[ns] into datetime'

    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime(1970, 1, 1) + timedelta(seconds=ts)





# TODO REFACTOR CON teslakit/database.py

def StoreBugXdset(xds_data, p_ncfile):
    # common
    from datetime import datetime, date
    import os
    import os.path as op

    # pip
    import netCDF4
    import numpy as np

    # tk
    from functions import npdt64todatetime as n2d

    '''
    Stores xarray.Dataset to .nc file while avoiding bug with time data (>2262)
    '''

    # get metadata from xarray.Dataset
    dim_names = xds_data.dims.keys()
    var_names = xds_data.variables.keys()

    # Handle time data  (calendar format)
    calendar = 'standard'
    units = 'hours since 1970-01-01 00:00:00'

    # remove previous file
    if op.isfile(p_ncfile):
        os.remove(p_ncfile)

    # Use netCDF4 lib
    root = netCDF4.Dataset(p_ncfile, 'w', format='NETCDF4')

    # Handle dimensions
    for dn in dim_names:
        vals = xds_data[dn].values[:]
        root.createDimension(dn, len(vals))

    # handle variables
    for vn in var_names:
        vals = xds_data[vn].values[:]

        # dimensions values
        if vn in dim_names:

            if vn == 'time':  # time dimension values
                # TODO: profile / acelerar
                if isinstance(vals[0], datetime):
                    pass

                elif isinstance(vals[0], date):
                    # TODO se pierde la resolucion horaria
                    # parse datetime.date to datetime.datetime
                    vals = [datetime.combine(d, datetime.min.time()) for d in vals]

                elif isinstance(vals[0], np.datetime64):
                    # parse numpy.datetime64 to datetime.datetime
                    vals = [n2d(d) for d in vals]

                dv = root.createVariable(varname=vn, dimensions=(vn,), datatype='int64')
                dv[:] = netCDF4.date2num(vals, units=units, calendar=calendar)
                dv.units = units
                dv.calendar = calendar

            else:
                dv = root.createVariable(varname=vn, dimensions=(vn,), datatype=type(vals[0]))
                dv[:] = vals

        # variables values
        else:
            vdims = xds_data[vn].dims
            vatts = xds_data[vn].attrs

            vv = root.createVariable(varname=vn,dimensions=vdims, datatype='float32')
            vv[:] = vals

            # variable attributes
            vv.setncatts(vatts)

    # global attributes
    root.setncatts(xds_data.attrs)

    # close file
    root.close()

