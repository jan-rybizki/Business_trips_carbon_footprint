import numpy as np
import time
from geopy.geocoders import Nominatim
import geopandas
from matplotlib import gridspec
import matplotlib.pyplot as plt

#Using Gregors code
# haversine formula to compute distance on a sphere, using mean Earth radius
def haversine(lat0,lon0,lat1,lon1):
    radius = 6371.0088 # mean Earth radius in km

    d_lat = np.radians(lat1-lat0)
    d_lon = np.radians(lon1-lon0)
    sin2 = (np.sin(d_lat/2.)**2 +
           np.cos(np.radians(lat0))*np.cos(np.radians(lat1))*np.sin(d_lon/2.)**2)
    unit_dist = 2.*np.arctan2(np.sqrt(sin2),np.sqrt(1.-sin2))
    
    return radius*unit_dist

# given a list of city names in <cities>, use geopy.geocoders Nominatim API to
#   request coordinates and 'best match' full location strings online
# returns three lists of: latitude, longitude, location-string
#   if a city was not found, return coordinates are set to NaN, and
#   and return strings to 'NA (original_name)'
def get_coordinates(cities):
    # instantiate a new Nominatim client to request location data online
    geo_app = Nominatim(user_agent="co2parsing")
    cities = np.array(cities,dtype=object)
    
    i_sort = np.argsort(cities)
    
    cities_sorted = cities[i_sort]
    
    # always include index 0, and include indices i where cities_sorted[i]!=cities_sorted[i-1]
    # to avoid requesting data for a city more than once
    i_check = np.concatenate(([0],np.where(cities_sorted[1:]!=cities_sorted[:-1])[0]+1),axis=None)
        
    lat_sorted=np.ndarray(cities_sorted.shape)
    lon_sorted=np.ndarray(cities_sorted.shape)
    name_sorted=np.ndarray(cities_sorted.shape,dtype=object)

    # assign coordinates and names found through geopy at indices <i_check>, to
    # only include unique city names once:
    k = 0
    for i in i_check:
        time.sleep(1.618)
        location_code = geo_app.geocode(cities_sorted[i])
        if location_code is None:
            lat_sorted[i] = float('NaN')
            lon_sorted[i] = float('NaN')
            name_sorted[i] = 'NA ('+cities_sorted[i]+')'
            continue
        # if the place was found, get location raw data:
        location = location_code.raw
        lat_sorted[i] = location['lat']
        lon_sorted[i] = location['lon']
        name_sorted[i] = location['display_name']
        k += 1
        if k%10 == 0:
            print(f"{i}/{len(cities_sorted)} cities processed")

    # assign coordinates and names to duplicate cities:
    for i in range(1,cities_sorted.shape[0]):
        if name_sorted[i] is not None: continue
        lat_sorted[i] = lat_sorted[i-1]
        lon_sorted[i] = lon_sorted[i-1]
        name_sorted[i] = name_sorted[i-1]
        
    lat=np.ndarray(cities_sorted.shape)
    lon=np.ndarray(cities_sorted.shape)
    name=np.ndarray(cities_sorted.shape,dtype=object)

    # reverse sorting:
    lat[i_sort] = lat_sorted
    lon[i_sort] = lon_sorted
    name[i_sort] = name_sorted
    
    return lat,lon,name

#cached Nominatim version from Antoine Goutenoir and Didier Barret
"""
import shelve
from core import get_path

class CachedGeocoder:

    def __init__(self, source="Nominatim", geocache="geocache.db"):
        self.geocoder = getattr(geopy.geocoders, source)()
        self.cache = shelve.open(get_path(geocache), writeback=True)
        # self.timestamp = time.time() + 1.5

    def geocode(self, address):
        if address not in self.cache:
            # time.sleep(max(0, 1 - (time.time() - self.timestamp)))
            time.sleep(1.618)
            # self.timestamp = time.time()
            self.cache[address] = self.geocoder.geocode(
                query=address,
                timeout=5,
                language='en_US',  # urgh
                addressdetails=True,  # only works with Nominatim /!.
            )
        return self.cache[address]

    def close(self):
        self.cache.close()
"""



# plot global map with travel destinations and monthly CO2eq emissions in a specific monthly interval
class canimatedmap:
    def __init__(self,lat,lon,times,durations,distances,co2eq,show_densities=False,fadeout_time=0.0, year0=2018):
        # some configuration variables
        # note: should be added as function arguments once stable
        # change color depending on density
        self.show_densities = show_densities
        # each point is visible for an additional fadeout_time months after return with decreasing opacity
        self.fadeout_time = float(fadeout_time)
        # increase point size by CO2eq
        self.vary_size_by_co2 = True
        # show straight arrows to destination (great circles would be cooler but I did not get to it)
        self.show_arrows = False
        
        self.x = lon
        self.y = lat
        self.t = times
        self.dt = durations
        self.d = distances
        self.eq = co2eq
        self.year0 = year0
        self.prepare_data()
        
    def set_fadeout_time(self,fadeout_time):
        self.fadeout_time=float(fadeout_time)
        
    def set_show_densities(self,show_densities):
        self.show_densities=show_densities
        
    def prepare_data(self):
        # (1) exclude NaN coordinates for plotting and
        # (2) remove place of origin from destinations by excluding zero distances
        cut = np.logical_or(np.logical_not(np.isnan(self.x)),self.d>0)
        self.x = self.x[cut]
        self.y = self.y[cut]
        self.t = self.t[cut]
        self.dt = self.dt[cut]
        self.d = self.d[cut]
        self.eq = self.eq[cut]

        # convert dt to months
        days_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=float)
        for i in range(12):
            cut = np.mod(self.t,12)==i
            self.dt[cut] = self.dt[cut]/days_in_month[i]

        # sort by return time for vanishing destinations (plot later over earlier points)
        idx = (self.t+self.dt).argsort()
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.t = self.t[idx]
        self.dt = self.dt[idx]
        self.d = self.d[idx]
        self.eq = self.eq[idx]
        
        self.n_months = self.t.max()+1 # number of months in the sequence

        # create monthly co2eq list:
        list_co2 = []
        for i in range(self.n_months):
            list_co2.append(self.eq[np.logical_and(self.t>=i,self.t<i+1)].sum())
        self.co2=np.array(list_co2)
            
        # set up x-ticks position  
        self.xtick_x = np.arange(self.n_months)

        # set up x-ticks label   
        xtick_label = xtick_label1 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        xtick_label[0] = str(self.year0)
        
        for i in range(self.n_months//12):
            xtick_label = xtick_label + xtick_label1
            xtick_label[12*(i+1)] = str(self.year0+i+1)
        
        self.xtick_label = np.array(xtick_label,dtype=object)
        self.xtick_label = xtick_label[:self.n_months]

        # set up world data for plotting
        self.world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        
    def get_months(self):
        return self.n_months
    
    def plot_scatter(self,ax,x1,y1,d1,eq1,alpha=1.0):
        if x1.size==0:
            return
        
        if self.show_densities:
            # get local density
            xy = np.vstack([x1,y1])
            if x1.shape[0]>1:
                try:
                    z1 = gaussian_kde(xy,bw_method=0.05)(xy)
                except np.linalg.linalg.LinAlgError:
                    z1 = np.zeros(x1.shape)
            else:
                z1 = np.zeros(x1.shape)
            # sort by density, and plot densest points last
            idx = z1.argsort()
            x1,y1,z1 = x1[idx],y1[idx],z1[idx]
            d1,eq1 = d1[idx],eq1[idx]
            c=z1
            # scale c from 0 to 1 (or set to 0 in case it is uniform)
            if(c.min()<c.max()):
                c=(c-c.min())/(c.max()-c.min())
            else:
                c=np.zeros(c.shape)
            # apply a colormap
            cmap = plt.cm.Reds
            c=cmap(1.0-0.5*c)
        else:
            c='#c00000'
            
        if self.vary_size_by_co2:
            s=48.0*eq1**0.5
        else:
            s=32.0

        ax.scatter(x1,y1,c=c,s=s,edgecolors='none',alpha=alpha)
        

    # stepsize is the stepsize in months used with fadeout_time to plot difference opacities over time
    def plot(self,month_0,month_1=-1.0,filename=None,stepsize=0.1,show_plot=True):
        if month_0 >= self.n_months+1.0:
            return False
        
        if month_1 < month_0:
            month_1 = month_0+1.0
                    
        fig = plt.figure(figsize=(12,8)) # "fig = " omitted
        gs = gridspec.GridSpec(2,1,height_ratios=[10,2])
        ax0 = plt.subplot(gs[0])
        # plot destination density
        self.world.plot(ax=ax0,color='#c6c6c6',edgecolor=None)
            
        # optionally display arrows in the map
        if self.show_arrows:
            x0 = lon0*np.ones(x1.shape)
            y0 = lat0*np.ones(y1.shape)
            for i in range(x1.shape[0]):
                ax0.arrow(x0[i],y0[i],x1[i]-x0[i],y1[i]-y0[i],head_width=1,head_length=1,fc='k',ec='k')
                
        # overlay scatter plot
        # set alpha to 1.0 if there is overlap of [t,t+dt] with [month_0,month_1[:
        cut = np.logical_and(self.t+self.dt>=month_0,self.t<month_1)
        self.plot_scatter(ax0,self.x[cut],self.y[cut],self.d[cut],self.eq[cut])
        # months up to the current month can be visible depending on the fadeout_time after return
        if self.fadeout_time>0:
            for m in np.arange(month_0-stepsize*np.ceil(self.fadeout_time/stepsize),month_0,stepsize):
                # m corresponds to the return time
                cut = np.logical_and(self.t+self.dt>=m,self.t+self.dt<m+stepsize)
                alpha = max( 0.0 , 0.4 * ( 1.0 - (month_0-m)/self.fadeout_time ) )
                self.plot_scatter(ax0,self.x[cut],self.y[cut],self.d[cut],self.eq[cut],alpha)
            
        ax0.axis(False)
        
        # plot CO2eq for each month up to month_1
        ax1 = plt.subplot(gs[1])
        ax1.set_ylim([0,self.co2.max()])
        ax1.set_xlim([-0.5,self.n_months-0.5])
        ax1.bar(self.xtick_x,np.select([self.xtick_x<int(np.ceil(month_1)),self.xtick_x>=int(np.ceil(month_1))],
                                       [self.co2,np.zeros(self.n_months)]),width=0.8,align='center')
        if month_0>0.0 or month_1<self.n_months:
            if month_1-month_0>0.8:
                width=month_1-month_0-0.2
            else:
                width=month_1-month_0
            ax1.bar(month_0-0.4,self.co2.max(),width=width,align='edge',color='#000000',alpha=0.2)
        ax1.set_xticks(self.xtick_x)
        ax1.set_xticklabels(self.xtick_label,rotation=45)
        ax1.set_ylabel('co2eq [t]')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show_plot:
            plt.show()
        plt.close(fig)