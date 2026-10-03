import sys
import numpy as np


class MapBackend:
    """Simple abstraction for Basemap (Python < 3.10) vs Cartopy (Python >= 3.10)."""
    
    @staticmethod
    def get():
        """Returns BasemapBackend for Python < 3.10, CartopyBackend otherwise."""
        if sys.version_info < (3, 10):
            return BasemapBackend()
        return CartopyBackend()


class BasemapBackend:
    """Basemap implementation for Python < 3.10."""
    
    def __init__(self):
        from mpl_toolkits.basemap import Basemap
        self.Basemap = Basemap
    
    def get_lcc_axes(self, fig):
        """Return axes for Lambert Conformal Conic projection."""
        return fig.add_subplot(122)
    
    def draw_ground_track(self, ax, latlons):
        """Draw great circle paths on LCC map."""
        m = self.Basemap(
            projection='lcc', lat_1=45., lat_2=55, lat_0=50, lon_0=-65.,
            resolution=None, width=9000000, height=9000000, ax=ax
        )
        for i in range(len(latlons) - 1):
            lon1, lat1 = float(latlons[i][1]), float(latlons[i][0])
            lon2, lat2 = float(latlons[i+1][1]), float(latlons[i+1][0])
            m.drawgreatcircle(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
        m.shadedrelief()
        m.drawparallels(np.arange(-90., 91., 30.))
        m.drawmeridians(np.arange(-180., 181., 60.))
        ax.set_title("Ground Track")
    
    def get_ortho_axes(self, fig, subplot, lon_0, lat_0):
        """Return axes for orthographic projection."""
        ax = fig.add_subplot(subplot)
        self._map = self.Basemap(
            projection='ortho', lon_0=lon_0, lat_0=lat_0,
            resolution='l', anchor='SW', ax=ax, suppress_ticks=False
        )
        return ax
    
    def draw_globe_features(self, ax):
        """Draw coastlines and land features."""
        self._map.drawcoastlines(ax=ax, linewidth=.25, zorder=5)
        self._map.fillcontinents(color='coral', lake_color='aqua', ax=ax, zorder=5)
        self._map.drawparallels(np.arange(-90., 120., 30.), ax=ax, zorder=5)
        self._map.drawmeridians(np.arange(0., 420., 60.), ax=ax, zorder=5)
        self._map.drawmapboundary(fill_color='aqua', ax=ax, zorder=2)


class CartopyBackend:
    """Cartopy implementation for Python >= 3.10."""
    
    def __init__(self):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from pyproj import Geod
        from matplotlib.patches import Rectangle
        from shapely.geometry import Polygon
        
        self.ccrs = ccrs
        self.cfeature = cfeature
        self.Geod = Geod
        self.Rectangle = Rectangle
        self.Polygon = Polygon
        self.globe = ccrs.Globe(ellipse=None, semimajor_axis=6370997, semiminor_axis=6370997)
        self._proj = None
    
    def get_lcc_axes(self, fig):
        """Return axes for Lambert Conformal Conic projection."""
        return fig.add_subplot(
            122,
            projection=self.ccrs.LambertConformal(
                central_latitude=50, central_longitude=-65
            )
        )
    
    def draw_ground_track(self, ax, latlons):
        """Draw great circle paths on LCC map."""
        proj = self.ccrs.LambertConformal(
            central_longitude=-65., central_latitude=50.,
            standard_parallels=(45., 55.), globe=self.globe
        )
        latlon = self.ccrs.PlateCarree(globe=self.globe)
        ax.set_extent([-3.8e6, 3.8e6, -4.4e6, 4.4e6], crs=proj)
        
        geod = self.Geod(a=6370997, b=6370997)
        for i in range(len(latlons) - 1):
            lon1, lat1 = float(latlons[i][1]), float(latlons[i][0])
            lon2, lat2 = float(latlons[i+1][1]), float(latlons[i+1][0])
            pts = geod.npts(lon1, lat1, lon2, lat2, 100)
            lons = np.array([lon1] + [p[0] for p in pts] + [lon2])
            lats = np.array([lat1] + [p[1] for p in pts] + [lat2])
            xyz = proj.transform_points(latlon, lons, lats)
            ax.plot(xyz[:, 0], xyz[:, 1])
        
        ax.stock_img()
        ax.gridlines(crs=latlon, xlocs=np.arange(-180., 181., 60.),
                     ylocs=np.arange(-90., 91., 30.), draw_labels=False,
                     color='k', linewidth=1.0, linestyle=(0, (1, 1)))
        ax.set_title("Ground Track")
    
    def get_ortho_axes(self, fig, subplot, lon_0, lat_0):
        """Return axes for orthographic projection."""
        self._proj = self.ccrs.Orthographic(
            central_longitude=lon_0, central_latitude=lat_0, globe=self.globe
        )
        ax = fig.add_subplot(subplot, projection=self._proj)
        ax.set_anchor('SW')
        return ax
    
    def draw_globe_features(self, ax):
        """Draw coastlines and land features."""
        ax.add_geometries(
            [self.Polygon(self._proj.boundary)], crs=self._proj,
            facecolor='aqua', edgecolor='none', zorder=3
        )
        ax.add_feature(self.cfeature.LAND, facecolor='coral', edgecolor='none', zorder=5)
        ax.add_feature(self.cfeature.LAKES, facecolor='aqua', edgecolor='none', zorder=5)
        ax.coastlines(resolution='110m', linewidth=.25, zorder=5)
        ax.gridlines(xlocs=np.arange(-180., 180., 60.),
                     ylocs=np.arange(-90., 120., 30.), draw_labels=False,
                     color='k', linewidth=1.0, linestyle=(0, (1, 1)), zorder=5)
