#Preamble
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
from multiprocessing import Pool
from multiprocessing import cpu_count
import lightkurve as lk
import math
import time
import copy
import pickle
import emcee
import corner
import os

os.environ["OMP_NUM_THREADS"] = "1"

def generateEllipse(a,b,centerX,centerY, grid, opacity):
    '''
    Creates a two dimensional array with an ellipse filled in
    Used by transitSim
    Parameters:
    a: The width of the ellipse
    b: The height of the ellipse
    centerX: The x coordinate of the center of the ellipse, must be larger than a
    centerY: The y coordinate of the center of the ellipse, mast be larger than b
    grid: The two dimennsional array to generate the ellipse on, must be larger than [centerX + a][centerY + b]
    opacity: Value to give the filled in values of the ellipse

    Returns:
    grid: the filled in grid with an ellipse
    '''
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if a > 0 and b > 0:
                if ((x-centerX)/a)*((x-centerX)/a) + ((y-centerY)/b)*((y-centerY)/b) <= 1:
                    grid[x][y] = opacity
    
    return grid

def transitSim(a,b,r, speed, times, tref, opacity, impact):
    '''
    Creates a light curve of a transiting ellipse from the paramters
    Used by logLikelihood and to plot the light curves
    Parameters:
    a: The width of the ellipse
    b: The height of the ellipse
    r: The radius of the star
    speed: The speed at which the ellipse transits (in stellar radii per unit time)
    times: A list of time points to generate a value for
    tref: The location of the center of the transit in the time array by value
    opacity: The opacity of the ellipse, although there is no technical limit for this value, the physical limit is [0,1] 
    impact: Impact parameter of the occulter
    '''
    ##Calculate the dimension ratios incase the ellipse has to be resized
    ab = a/b
    br = b/r
    
    tmin = tref - 1/(2*speed) ##Calculate the lower time bound for the transit
    tmax = tref + 1/(2*speed) ##Calculate the upper time bound for the transit
    transitTimes = [t for t in times if t >= tmin and t <= tmax] ##Create an array of only the times included in the transit
    flux = [1 for t in times if t < tmin] ##Create an array of the flux values, setting it to 1 before the transit
    differences = [] ##Calculate the differences between the time points
    for i in range(1,len(transitTimes)):
        differences.append(transitTimes[i] - transitTimes[i-1])
    
    if len(differences) == 0:
        differences.append(times[1] - times[0])
    minDiff = np.min(differences)
    intDiffs = [int(d/minDiff) for d in differences]##Normalize the difference so the smallest one has value unity
    length = np.sum(intDiffs)##Calculate the length of the transit in steps
    
    
    res = int((2*(b+r))/length) ##Resize the dimensions so the length matches what is required for the time array
    if res < 1:
        res = 1
    rnew = int(res*length/(2*(1+br)))
    bnew = int(br*rnew)
    anew = int(ab*bnew)
    newIntDiffs = [res*i for i in intDiffs]
    
    starGrid = np.zeros([2*rnew+impact, 4*bnew+2*rnew+4])
    ellipseGrid = np.zeros([2*rnew+impact, 4*bnew+2*rnew+4])
    starGrid = generateEllipse(rnew,rnew,rnew, 2*bnew+rnew, starGrid,1)##Generate the star
    ellipseGrid = generateEllipse(anew,bnew,rnew+impact,3*bnew+2*rnew + 2,ellipseGrid,opacity)##Genrate the occulter
    planetGrid = np.ones([2*rnew+impact,4*bnew+2*rnew+ 4]) - ellipseGrid
    fluxGrid = np.multiply(starGrid,planetGrid)##Calculate the flux by item by item multiplying each pixel in the star and planet grid and summing
    initialFlux = np.sum(fluxGrid)
    
    for i in newIntDiffs:
        for j in range(i):
            planetGrid = np.delete(planetGrid,0,1)
            planetGrid = np.append(planetGrid,np.ones([2*rnew+impact,1]),1)##Move the first column to the end to "move" the planet across the star
            
        
        fluxGrid = np.multiply(starGrid,planetGrid)
        percentFlux = np.sum(fluxGrid)/initialFlux
        flux.append(percentFlux)#Calculate a normalized value of flux
        
    for t in times:
        if t > tmax:
            flux.append(1)##Add full flux after the transit
    
    flux.append(1)
    return flux

def logLikelihood(theta, times,tRef, flux, fluxErr):
    """
    Calculates the log likelihood based on the difference between the model and the data
    Used by logProbability
    Args:
        theta (list) - parameters of the model
        times (list) - time array of the light curve
        flux (list) - array of flux data points
        fluxErr (list) - array of errors for the flux data points

    Returns:
        lnl (float) - log likelihood for the given theta values
    """
    xdim, ydim, velocity, opacity, impact = theta
    fluxPredicted = transitSim(ydim, xdim,50,velocity,times, tRef,opacity,int(impact))
    error = [((flux[i] - fluxPredicted[i])**2) /(2*fluxErr[i]**2) for i in range(len(flux))]
    lnl = -np.sum(error)
    return lnl

def logPrior(theta, times, minFlux):
    """
    Returns flat priors, checking that the given theta values are physically possible
    Used by logProbability
    Args:
        theta (list) - parameters of the model
        times (list) time array of the light curve
        
    Returns: 
        lnPrior (float) - fixed log prior value if theta values are allowed, -inf if theta values aren't
    """
    xdim, ydim, velocity, opacity,impact = theta
    lnPrior = 0
    if 0 < xdim < 50 and 0 < ydim < 50 and 0 < velocity < 50 and 0 < opacity < 1.01 and 0 < impact < 100: ##Check to see if the shape exists but is not larger than the star
        ##Also check to see that it transits in a consistent direction and not extremely fast and that it is not super dark
        lnPrior +=  + np.log(1/50) + np.log(1)
    else:
        return -np.inf

    predictedRadius = 50*np.sqrt(1-minFlux)
    lnPrior += (5000/(np.sqrt(2*3.1415926535)*5))*np.exp(-0.5*((xdim-predictedRadius)/5)**2)
    lnPrior += (5000/(np.sqrt(2*3.1415926535)*5))*np.exp(-0.5*((ydim-predictedRadius)/5)**2)
    return lnPrior

def logProbability(theta, times, tRef, flux, fluxErr,minFlux):
    """
    Combines the log likelihood and log prior to get log probability
    Used by emcee.sample()
    Args:
        theta (list) - parameters of the model
        times (list)
    """
    startTime = time.time()
    lp = logPrior(theta, times,minFlux)
    if not np.isfinite(lp):
        return -np.inf
    ll = logLikelihood(theta, times,tRef, flux, fluxErr)
    endTime = time.time()
    #print(endTime - startTime)
    return (lp + ll)






'''
def load_lc(fp, fluxtype="PDC", mask=False):
    """Load light curve data from pickle file into a lightkurve object
    Args:
        fp (str) - file path to pickle file in standard format
        fluxtype (str) - Type of flux to prioritize,
            choose between "raw", "corr", and "PDC"
        mask (bool) - Mask data points non-zero flags in quality

    returns:
        lc (lightkurve.lightcurve.LightCurve) - a LightCurve object
    """

    with open(fp, 'rb') as file:
        lc_list = pickle.load(file)

    fluxes = {"raw": lc_list[7], "corr": lc_list[8], "PDC": lc_list[9]}

    try:
        flux = fluxes[fluxtype]

    except KeyError:
        print("""
        The flux type must be 'raw', 'corr', or 'PDC'. Defaulting to 'PDC'.""")
        flux = fluxes["PDC"]

    finally:
        time = lc_list[6]
        flux_err = lc_list[10]
        quality = lc_list[11]

        if mask:
            mask = lc_list[11] == 0
            flux = flux[mask]
            time = time[mask]
            flux_err = flux_err[mask]
            quality = quality[mask]  # just 0's if masked

        # for meta information
        fluxes.update(
            {"TESS Magnitude": lc_list[3], "filename": fp.split("/")[-1]})
        lc = lk.lightcurve.TessLightCurve(
            time=time, flux=flux, flux_err=flux_err, targetid=lc_list[0],
            quality=quality, camera=lc_list[4], ccd=lc_list[5],
            ra=lc_list[1], dec=lc_list[2], label=f"TIC {lc_list[0]}",
            meta=fluxes
        )

    return lc
'''

'''
##Code for pulling a light curve from lightkurve
searchResult = lk.search_lightcurve("KIC 8462852", quarter = 8)
lc = searchResult.download()
'''
##lc = load_lc("./lcs/tesslc_67646988.pkl")
searchResult = lk.search_lightcurve("TIC 200297691")
lc = searchResult.download()
lc = lc.flatten()
lc = lc.normalize()
lc = lc.remove_nans()
lc = lc.remove_nans('flux_err')
#lc = lc.truncate(57170,57175)
times = lc.time.mjd
flux = lc.flux.value
fluxErr = lc.flux_err.value

##Identify the minimum value flux to use as the center of the transit
minFlux = np.min(flux)
index = np.where(flux == minFlux)
tRef = times[index[0]]
print(tRef)
lc = lc.truncate(tRef-3,tRef+3)



ncpu = cpu_count()
print("{0} CPUs".format(ncpu))





#Make a guess at the light curve
pos = [15,15,6,1,50] * np.ones([20,5]) + [15,15,6,1,50]*((np.random.random([20,5])-0.5)/5)
nwalkers, ndim = pos.shape

##Sample!!!!
print("Beginning Sampling")
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args = (times, tRef, flux, fluxErr,minFlux), pool = pool)
    sampler.run_mcmc(pos,8000)




##This block plots the traces of the 5parameters
fig, axes = plt.subplots(6,figsize = (10,10), sharex = True)
samples = sampler.get_chain()
logProb = sampler.get_log_prob()
labels = ["X dim", "Y dim", "Pixel Speed", "Opacity", "Impact Parameter"]
for i in range(ndim):
    axes[i].plot(samples[:,:,i],'k',alpha = 0.3)
    axes[i].set_xlim(0,len(samples))
    axes[i].set_ylabel(labels[i])
    
axes[-1].plot(logProb[:,:],'k',alpha = 0.3)
axes[-1].set_xlim(0,len(samples))
axes[-1].set_ylabel("Log Probability")
axes[-1].set_xlabel("Step Number")
plt.savefig("tracesTIC200297691.png")

fig = plt.subplot()




##This section takes 100 of the samples from the later 80% and plots then over the true curve
flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    try:
        sample = flat_samples[ind]
        sampleFlux = transitSim(sample[0],sample[1],50,sample[2],times,tRef,sample[3],int(sample[4]))
        fig.plot(times,sampleFlux,alpha = 0.1, color = 'orange')
    except:
        print("Invalid parameters", sample)
    

fig.plot(times,flux, alpha = 0.5, color = 'blue',ls = '-', marker = 'o')
plt.savefig("transitsTIC200297691.png")




##Make a corner plot
flatSamples = sampler.get_chain(discard = 500, flat = True)
fig = corner.corner(flatSamples, bins = 40, labels = labels)
plt.savefig("cornerTIC200297691.png")

for i in range(ndim):
    print(np.mean(samples[:,:,i]),np.std(samples[:,:,i]))