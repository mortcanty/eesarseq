# application.py
# widget interface SAR for sequential change detection
# colab version

import ee
ee.Initialize
import time, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gamma, f, chi2
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,
                        basemaps,basemap_to_tiles,
                        LayersControl)
from geopy.geocoders import Nominatim

# *****************************************
# The sequental change detection algorithm
# *****************************************

def chi2cdf(chi2, df):
    """Calculates Chi square cumulative distribution function for
       df degrees of freedom using the built-in incomplete gamma
       function gammainc().
    """
    return ee.Image(chi2.divide(2)).gammainc(ee.Number(df).divide(2))

def det(im):
    """Calculates determinant of 2x2 diagonal covariance matrix."""
    return im.expression('b(0)*b(1)')

def log_det_sum(im_list, j):
    """Returns log of determinant of the sum of the first j images in im_list."""
    im_ist = ee.List(im_list)
    sumj = ee.ImageCollection(im_list.slice(0, j)).reduce(ee.Reducer.sum())
    return ee.Image(det(sumj)).log()

def log_det(im_list, j):
    """Returns log of the determinant of the jth image in im_list."""
    im = ee.Image(ee.List(im_list).get(j.subtract(1)))
    return ee.Image(det(im)).log()

def pval(im_list, j, m=4.4):
    """Calculates -2logRj for im_list and returns P value and -2logRj."""
    im_list = ee.List(im_list)
    j = ee.Number(j)
    m2logRj = (log_det_sum(im_list, j.subtract(1))
               .multiply(j.subtract(1))
               .add(log_det(im_list, j))
               .add(ee.Number(2).multiply(j).multiply(j.log()))
               .subtract(ee.Number(2).multiply(j.subtract(1))
               .multiply(j.subtract(1).log()))
               .subtract(log_det_sum(im_list,j).multiply(j))
               .multiply(-2).multiply(m))
    pv = ee.Image.constant(1).subtract(chi2cdf(m2logRj, 2))
    return (pv, m2logRj)

def p_values(im_list):
    """Pre-calculates the P-value array for a list of images."""
    im_list = ee.List(im_list)
    k = im_list.length()

    def ells_map(ell):
        """Arranges calculation of pval for combinations of k and j."""
        ell = ee.Number(ell)
        # Slice the series from k-l+1 to k (image indices start from 0).
        im_list_ell = im_list.slice(k.subtract(ell), k)

        def js_map(j):
            """Applies pval calculation for combinations of k and j."""
            j = ee.Number(j)
            pv1, m2logRj1 = pval(im_list_ell, j)
            return ee.Feature(None, {'pv': pv1, 'm2logRj': m2logRj1})

        # Map over j=2,3,...,l.
        js = ee.List.sequence(2, ell)
        pv_m2logRj = ee.FeatureCollection(js.map(js_map))

        # Calculate m2logQl from collection of m2logRj images.
        m2logQl = ee.ImageCollection(pv_m2logRj.aggregate_array('m2logRj')).sum()
        pvQl = ee.Image.constant(1).subtract(chi2cdf(m2logQl, ell.subtract(1).multiply(2)))
        pvs = ee.List(pv_m2logRj.aggregate_array('pv')).add(pvQl)
        return pvs

    # Map over l = k to 2.
    ells = ee.List.sequence(k, 2, -1)
    pv_arr = ells.map(ells_map)

    # Return the P value array ell = k,...,2, j = 2,...,l.
    return pv_arr

def filter_j(current, prev):
    """Calculates change maps; iterates over j indices of pv_arr."""
    pv = ee.Image(current)
    prev = ee.Dictionary(prev)
    pvQ = ee.Image(prev.get('pvQ'))
    i = ee.Number(prev.get('i'))
    cmap = ee.Image(prev.get('cmap'))
    smap = ee.Image(prev.get('smap'))
    fmap = ee.Image(prev.get('fmap'))
    bmap = ee.Image(prev.get('bmap'))
    alpha = ee.Image(prev.get('alpha'))
    j = ee.Number(prev.get('j'))
    cmapj = cmap.multiply(0).add(i.add(j).subtract(1))
    # Check      Rj?            Ql?                  Row i?
    tst = pv.lt(alpha).And(pvQ.lt(alpha)).And(cmap.eq(i.subtract(1)))
    # Then update cmap...
    cmap = cmap.where(tst, cmapj)
    # ...and fmap...
    fmap = fmap.where(tst, fmap.add(1))
    # ...and smap only if in first row.
    smap = ee.Algorithms.If(i.eq(1), smap.where(tst, cmapj), smap)
    # Create bmap band and add it to bmap image.
    idx = i.add(j).subtract(2)
    tmp = bmap.select(idx)
    bname = bmap.bandNames().get(idx)
    tmp = tmp.where(tst, 1)
    tmp = tmp.rename([bname])
    bmap = bmap.addBands(tmp, [bname], True)
    return ee.Dictionary({'i': i, 'j': j.add(1), 'alpha': alpha, 'pvQ': pvQ,
                          'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap':bmap})

def filter_i(current, prev):
    """Arranges calculation of change maps; iterates over row-indices of pv_arr."""
    current = ee.List(current)
    pvs = current.slice(0, -1 )
    pvQ = ee.Image(current.get(-1))
    prev = ee.Dictionary(prev)
    i = ee.Number(prev.get('i'))
    alpha = ee.Image(prev.get('alpha'))
    median = prev.get('median')
    # Filter Ql p value if desired.
    pvQ = ee.Algorithms.If(median, pvQ.focal_median(2.5), pvQ)
    cmap = prev.get('cmap')
    smap = prev.get('smap')
    fmap = prev.get('fmap')
    bmap = prev.get('bmap')
    first = ee.Dictionary({'i': i, 'j': 1, 'alpha': alpha ,'pvQ': pvQ,
                           'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap': bmap})
    result = ee.Dictionary(ee.List(pvs).iterate(filter_j, first))
    return ee.Dictionary({'i': i.add(1), 'alpha': alpha, 'median': median,
                          'cmap': result.get('cmap'), 'smap': result.get('smap'),
                          'fmap': result.get('fmap'), 'bmap': result.get('bmap')})
    
def dmap_iter(current, prev):
    """Reclassifies values in directional change maps."""
    prev = ee.Dictionary(prev)
    j = ee.Number(prev.get('j'))
    image = ee.Image(current)
    avimg = ee.Image(prev.get('avimg'))
    diff = image.subtract(avimg)
    # Get positive/negative definiteness.
    posd = ee.Image(diff.select(0).gt(0).And(det(diff).gt(0)))
    negd = ee.Image(diff.select(0).lt(0).And(det(diff).gt(0)))
    bmap = ee.Image(prev.get('bmap'))
    bmapj = bmap.select(j)
    dmap = ee.Image.constant(ee.List.sequence(1, 3))
    bmapj = bmapj.where(bmapj, dmap.select(2))
    bmapj = bmapj.where(bmapj.And(posd), dmap.select(0))
    bmapj = bmapj.where(bmapj.And(negd), dmap.select(1))
    bmap = bmap.addBands(bmapj, overwrite=True)
    # Update avimg with provisional means.
    i = ee.Image(prev.get('i')).add(1)
    avimg = avimg.add(image.subtract(avimg).divide(i))
    # Reset avimg to current image and set i=1 if change occurred.
    avimg = avimg.where(bmapj, image)
    i = i.where(bmapj, 1)
    return ee.Dictionary({'avimg': avimg, 'bmap': bmap, 'j': j.add(1), 'i': i})

def change_maps(im_list, median=False, alpha=0.01):
    """Calculates thematic change maps."""
    k = im_list.length()
    # Pre-calculate the P value array.
    pv_arr = ee.List(p_values(im_list))
    # Filter P values for change maps.
    cmap = ee.Image(im_list.get(0)).select(0).multiply(0)
    bmap = ee.Image.constant(ee.List.repeat(0,k.subtract(1))).add(cmap)
    alpha = ee.Image.constant(alpha)
    first = ee.Dictionary({'i': 1, 'alpha': alpha, 'median': median,
                           'cmap': cmap, 'smap': cmap, 'fmap': cmap, 'bmap': bmap})
    result = ee.Dictionary(pv_arr.iterate(filter_i, first))
    # Post-process bmap for change direction.
    bmap =  ee.Image(result.get('bmap'))
    avimg = ee.Image(im_list.get(0))
    j = ee.Number(0)
    i = ee.Image.constant(1)
    first = ee.Dictionary({'avimg': avimg, 'bmap': bmap, 'j': j, 'i': i})
    dmap = ee.Dictionary(im_list.slice(1).iterate(dmap_iter, first)).get('bmap')
    return ee.Dictionary(result.set('bmap', dmap))

# ********************
# The widget interface
# ********************

poly = None
geolocator = Nominatim(timeout=10,user_agent='tutorial-pt-4.ipynb')

w_location = widgets.Text(
    value='Jülich',
    placeholder=' ',
    description='',
    disabled=False
)
w_orbitpass = widgets.RadioButtons(
    options=['ASCENDING','DESCENDING'],
    value='ASCENDING',
    description='Pass:',
    disabled=False
)
w_changemap = widgets.RadioButtons(
    options=['Bitemporal','First','Last','Frequency'],
    value='First',
    disabled=False
)
w_interval = widgets.BoundedIntText(
    min=1,
    value=1,
    description='Interval:',
    disabled=True
)
w_maxfreq = widgets.BoundedIntText(
    min=1,
    value=20,
    description='MaxFrequency:',
    disabled=True
)
w_platform = widgets.RadioButtons(
    options=['Both','A','B'],
     value='Both',
    description='Platform:',
    disabled=False
)
w_relativeorbitnumber = widgets.IntText(
    value='0',
    description='Rel orbit:',
    disabled=False
)
w_exportassetsname = widgets.Text(
    value='projects/<username>/assets/<path>',
    placeholder=' ',
    disabled=False
)
w_exportdrivename = widgets.Text(
    value='<path>',
    placeholder=' ',
    disabled=False
)
w_exportscale = widgets.FloatText(
    value=10,
    placeholder=' ',
    description='Scale ',
    disabled=False
)
w_startdate = widgets.Text(
    value='2018-04-01',
    placeholder=' ',
    description='Start date:',
    disabled=False
)
w_enddate = widgets.Text(
    value='2018-11-01',
    placeholder=' ',
    description='End date:',
    disabled=False
)
w_stride = widgets.BoundedIntText(
    value=1,
    min=1,
    description='Stride:',
    disabled=False
)
w_median = widgets.Checkbox(
    value=True,
    description='5x5 Median filter',
    disabled=False
)
w_quick = widgets.Checkbox(
    value=True,
    description='Quick Preview',
    disabled=False
)
w_significance = widgets.BoundedFloatText(
    value='0.01',
    min=0.001,
    max=0.05,
    step=0.001,
    description='Signif:',
    disabled=False
)
w_maskchange = widgets.Checkbox(
    value=False,
    description='NC mask',
    disabled=False
)
w_maskwater = widgets.Checkbox(
    value=True,
    description='Water mask',
    disabled=False
)
w_opacity = widgets.BoundedFloatText(
    value='1.0',
    min=0.0,
    max=1.0,
    step=0.1,
    description='Opacity:',
    disabled=False
)
w_out = widgets.Output(
    layout=widgets.Layout(border='1px solid black')
)

w_collect = widgets.Button(description="Collect",disabled=True)
w_preview = widgets.Button(description="Preview",disabled=True)
w_review = widgets.Button(description="ReviewAsset",disabled=False)
w_reset = widgets.Button(description='Clear',disabled=False)
w_goto = widgets.Button(description='GoTo',disabled=False)
w_export_ass = widgets.Button(description='ExportToAssets',disabled=True)
w_export_drv = widgets.Button(description='ExportToDrive',disabled=True)
w_plot = widgets.Button(description='PlotFromAsset',disabled=False)

w_masks = widgets.VBox([w_maskchange,w_maskwater,w_quick])
w_dates = widgets.VBox([w_startdate,w_enddate])
w_assets = widgets.VBox([w_review,w_plot])
w_bmap = widgets.VBox([w_interval,w_maxfreq])
w_export = widgets.VBox([widgets.HBox([w_export_ass,w_exportassetsname]),widgets.HBox([w_export_drv,w_exportdrivename])])
w_signif = widgets.VBox([w_significance,w_median])

def on_widget_change(b):
    w_preview.disabled = True
    w_export_ass.disabled = True
    w_export_drv.disabled = True

def on_changemap_widget_change(b):   
    if b['new']=='Bitemporal':
        w_interval.disabled=False
    else:
        w_interval.disabled=True    
    if b['new']=='Frequency':
        w_maxfreq.disabled=False
    else:
        w_maxfreq.disabled=True    

def on_reset_button_clicked(b):
    with w_out:
        w_out.clear_output()
        print('Algorithm output')   
        
w_reset.on_click(on_reset_button_clicked)       

def on_goto_button_clicked(b):
    try:
        location = geolocator.geocode(w_location.value)
        m.center = (location.latitude,location.longitude)
        m.zoom = 11
    except Exception as e:
        with w_out:
            print('Error: %s'%e)

w_goto.on_click(on_goto_button_clicked)

#These widget changes require a new collect
w_orbitpass.observe(on_widget_change,names='value')
w_platform.observe(on_widget_change,names='value')
w_relativeorbitnumber.observe(on_widget_change,names='value')
w_startdate.observe(on_widget_change,names='value')
w_enddate.observe(on_widget_change,names='value')
w_stride.observe(on_widget_change,names='value')
w_median.observe(on_widget_change,names='value')
w_significance.observe(on_widget_change,names='value')
w_changemap.observe(on_changemap_widget_change,names='value')  

row1 = widgets.HBox([w_platform,w_orbitpass,w_relativeorbitnumber,w_dates],layout=widgets.Layout(border='1px solid black'))
row2 = widgets.HBox([w_collect,w_signif,w_stride,w_export,w_assets],layout=widgets.Layout(border='1px solid black'))
row3 = widgets.HBox([w_preview,w_changemap,w_bmap,w_masks,w_opacity],layout=widgets.Layout(border='1px solid black'))
row4 = widgets.HBox([w_reset,w_out,w_goto,w_location],layout=widgets.Layout(border='1px solid black'))

box = widgets.VBox([row1,row2,row3,row4])

#@title Collect

def get_incidence_angle(image):
    ''' grab the mean incidence angle '''
    result = ee.Image(image).select('angle') \
           .reduceRegion(ee.Reducer.mean(),geometry=poly,maxPixels=1e9) \
           .get('angle') \
           .getInfo()
    if result is not None:
        return round(result,2)
    else:
        #incomplete overlap, so use all of the image geometry        
        return round(ee.Image(image).select('angle') \
           .reduceRegion(ee.Reducer.mean(),maxPixels=1e9) \
           .get('angle') \
           .getInfo(),2)
        
def GetTileLayerUrl(image):
    map_id = ee.Image(image).getMapId()
    return map_id["tile_fetcher"].url_format        

def handle_draw(self, action, geo_json):
    global poly
    coords =  geo_json['geometry']['coordinates']
    if action == 'created':
        poly = ee.Geometry.Polygon(coords)
        w_preview.disabled = True
        w_export_ass.disabled = True
        w_export_drv.disabled = True 
        w_collect.disabled = False
    elif action == 'deleted':
        poly = None
        w_collect.disabled = True  
        w_preview.disabled = True    
        w_export_ass.disabled = True
        w_export_drv.disabled = True      

def getS1collection():
    s1 =  ee.ImageCollection('COPERNICUS/S1_GRD') \
                      .filterBounds(poly) \
                      .filterDate(ee.Date(w_startdate.value), ee.Date(w_enddate.value)) \
                      .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV','VH'])) \
                      .filter(ee.Filter.eq('resolution_meters', 10)) \
                      .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                      .filter(ee.Filter.eq('orbitProperties_pass', w_orbitpass.value))    
    return s1.filter(ee.Filter.contains(rightValue=poly,leftField='.geo'))

def get_vvvh(image):   
    ''' get 'VV' and 'VH' bands from sentinel-1 imageCollection and restore linear signal from db-values '''
    return image.select('VV','VH').multiply(ee.Image.constant(math.log(10.0)/10.0)).exp()

def clipList(current,prev):
    ''' clip a list of images and multiply by ENL'''
    imlist = ee.List(ee.Dictionary(prev).get('imlist'))
    poly = ee.Dictionary(prev).get('poly') 
    enl = ee.Number(ee.Dictionary(prev).get('enl')) 
    ctr = ee.Number(ee.Dictionary(prev).get('ctr'))   
    stride = ee.Number(ee.Dictionary(prev).get('stride'))
    imlist =  ee.Algorithms.If(ctr.mod(stride).eq(0),
        imlist.add(ee.Image(current).multiply(enl).clip(poly)),imlist)
    return ee.Dictionary({'imlist':imlist,'poly':poly,'enl':enl,'ctr':ctr.add(1),'stride':stride})    

def on_collect_button_clicked(b):
    ''' Collect a time series from the archive 
    '''
    global result, count, timestamplist1, archive_crs
    with w_out:
        try:
            w_out.clear_output()
            print('running on GEE archive COPERNICUS/S1_GRD (please wait for raster overlay) ...')
            collection = getS1collection()             
            if w_relativeorbitnumber.value > 0:
                collection = collection.filter(ee.Filter.eq('relativeOrbitNumber_start', int(w_relativeorbitnumber.value)))   
            if w_platform.value != 'Both':
                collection = collection.filter(ee.Filter.eq('platform_number', w_platform.value))         
            collection = collection.sort('system:time_start') 
            acquisition_times = ee.List(collection.aggregate_array('system:time_start')).getInfo()              
            count = len(acquisition_times)      
            if count<2:
                raise ValueError('Less than 2 images found')
            archive_crs = ee.Image(collection.first()).select(0).projection().crs().getInfo() 
            timestamplist = []
            for timestamp in acquisition_times:
                tmp = time.gmtime(int(timestamp)/1000)
                timestamplist.append(time.strftime('%x', tmp))  
            #Make timestamps in YYYYMMDD format            
            timestamplist = [x.replace('/','') for x in timestamplist]  
            timestamplist = ['T20'+x[4:]+x[0:4] for x in timestamplist]         
            timestamplist = timestamplist[::int(w_stride.value)]
            #In case of duplicates add running integer
            timestamplist1 = [timestamplist[i] + '_' + str(i+1) for i in range(len(timestamplist))]     
            count = len(timestamplist)
            if count<2:
                raise ValueError('Less than 2 images found, decrease stride')            
            relativeorbitnumbers = map(int,ee.List(collection.aggregate_array('relativeOrbitNumber_start')).getInfo())
            rons = list(set(relativeorbitnumbers))
            print('Images found: %i, platform: %s'%(count,w_platform.value))
            print('Number of 10m pixels contained: %i'%math.floor(poly.area().getInfo()/100.0))
            print('Acquisition dates: %s to %s'%(str(timestamplist[0]),str(timestamplist[-1])))
            print('Relative orbit numbers: '+str(rons))
            if len(rons)==1:
                mean_incidence = get_incidence_angle(collection.first())
                print('Mean incidence angle: %f'%mean_incidence)
            else:
                mean_incidence = 'undefined'
                print('Mean incidence angle: (select one rel. orbit)')
            pcollection = collection.map(get_vvvh)            
            collectionfirst = ee.Image(pcollection.first())
            w_exportscale.value = collectionfirst.projection().nominalScale().getInfo()          
            pList = pcollection.toList(500)   
            first = ee.Dictionary({'imlist':ee.List([]),'poly':poly,'enl':ee.Number(4.4),'ctr':ee.Number(0),'stride':ee.Number(int(w_stride.value))}) 
            imList = ee.List(ee.Dictionary(pList.iterate(clipList,first)).get('imlist'))              
            #Get a preview as collection mean                                           
            collectionmosaic = collection.mosaic().select(0,1).rename('b0','b1')
            percentiles = collectionmosaic.reduceRegion(ee.Reducer.percentile([2,98]),geometry=poly,scale=w_exportscale.value,maxPixels=10e9)
            mn = ee.Number(percentiles.get('b0_p2'))
            mx = ee.Number(percentiles.get('b0_p98'))        
            vorschau = collectionmosaic.select(0).visualize(min=mn, max=mx, opacity=w_opacity.value) 
            #Run the algorithm ************************************************
            result = change_maps(imList, w_median.value, w_significance.value)
            #******************************************************************
            w_preview.disabled = False
            w_export_ass.disabled = False
            w_export_drv.disabled = False
            #Display preview 
            if len(m.layers)>3:
                m.remove_layer(m.layers[3])
            m.add_layer(TileLayer(url=GetTileLayerUrl(vorschau)))                
        except Exception as e:
            print('Error: %s'%e) 

w_collect.on_click(on_collect_button_clicked)                  

#@title Preview

watermask = ee.Image('UMD/hansen/global_forest_change_2015').select('datamask').eq(1)  

def on_preview_button_clicked(b):
    ''' Preview change maps
    '''
    with w_out:  
        try:       
            jet = 'black,blue,cyan,yellow,red'
            rcy = 'black,red,cyan,yellow'
            palette = jet
            w_out.clear_output()
            print('Series length: %i images, previewing (please wait for raster overlay) ...'%count)
            if w_changemap.value=='First':
                mp = ee.Image(result.get('smap')).byte()
                mx = count
                print('Interval of first change:\n blue = early, red = late')
            elif w_changemap.value=='Last':
                mp=ee.Image(result.get('cmap')).byte()
                mx = count
                print('Interval of last change:\n blue = early, red = late')
            elif w_changemap.value=='Frequency':
                mp = ee.Image(result.get('fmap')).byte()
                mx = w_maxfreq.value
                print('Change frequency :\n blue = few, red = many')
            else:
                sel = int(w_interval.value)
                sel = min(sel,count-1)
                sel = max(sel,1)
                print('Bitemporal: %s-->%s'%(timestamplist1[sel-1],timestamplist1[sel]))
                print('red = positive definite, cyan = negative definite, yellow = indefinite')  
                bmap = ee.Image(result.get('bmap')).byte()   
                mp = bmap.select(sel-1).clip(poly)
                palette = rcy
                mx = 3     
            if len(m.layers)>3:
                m.remove_layer(m.layers[3])
            if not w_quick.value:
                mp = mp.reproject(crs=archive_crs,scale=float(w_exportscale.value))
            if w_maskwater.value==True:
                mp = mp.updateMask(watermask)
            if w_maskchange.value==True:    
                mp = mp.updateMask(mp.gt(0))    
            m.add_layer(TileLayer(url=GetTileLayerUrl(mp.visualize(min=0, max=mx, opacity=w_opacity.value, palette=palette))))
        except Exception as e:
            print('Error: %s'%e)

w_preview.on_click(on_preview_button_clicked)      

def on_review_button_clicked(b):
    ''' Examine change maps exported to user's assets
    ''' 
    with w_out:  
        try: 
#          test for existence of asset                  
            _ = ee.Image(w_exportassetsname.value).getInfo()
#          ---------------------------            
            asset = ee.Image(w_exportassetsname.value)
            poly = ee.Geometry.Polygon(ee.Geometry(asset.get('system:footprint')).coordinates())
            center = poly.centroid().coordinates().getInfo()
            center.reverse()
            m.center = center  
            bnames = asset.bandNames().getInfo()[3:-2]
            count = len(bnames)               
            jet = 'black,blue,cyan,yellow,red'
            rcy = 'black,red,cyan,yellow'
            smap = asset.select('smap').byte()
            cmap = asset.select('cmap').byte()
            fmap = asset.select('fmap').byte()
            bmap = asset.select(list(range(3,count+3)),bnames).byte()      
            palette = jet
            w_out.clear_output()
            print('Series length: %i images, reviewing (please wait for raster overlay) ...'%(count+1))
            if w_changemap.value=='First':
                mp = smap
                mx = count
                print('Interval of first change:\n blue = early, red = late')
            elif w_changemap.value=='Last':
                mp = cmap
                mx = count
                print('Interval of last change:\n blue = early, red = late')
            elif w_changemap.value=='Frequency':
                mp = fmap
                mx = w_maxfreq.value
                print('Change frequency :\n blue = few, red = many')
            else:
                sel = int(w_interval.value)-1
                sel = min(sel,count-1)
                sel = max(sel,0)
                if sel>0:
                    print('Bitemporal: %s --> %s'%(bnames[sel-1],bnames[sel]))
                else:
                    print('Bitemporal: image1 --> %s'%bnames[sel])
                print('red = positive definite, cyan = negative definite, yellow = indefinite')     
                mp = bmap.select(sel)
                palette = rcy
                mx = 3     
            if len(m.layers)>3:
                m.remove_layer(m.layers[3])
            if w_maskwater.value==True:
                mp = mp.updateMask(watermask)
            if w_maskchange.value==True:    
                mp = mp.updateMask(mp.gt(0))    
            m.add_layer(TileLayer(url=GetTileLayerUrl(mp.visualize(min=0, max=mx, opacity=w_opacity.value, palette=palette))))
            w_collect.disabled = False
        except Exception as e:
            print('Error: %s'%e)
    
w_review.on_click(on_review_button_clicked)   

def on_export_ass_button_clicked(b):
    ''' Export to assets
    '''
    try:
        smap = ee.Image(result.get('smap')).byte()
        cmap = ee.Image(result.get('cmap')).byte()
        fmap = ee.Image(result.get('fmap')).byte() 
        bmap = ee.Image(result.get('bmap')).byte()                    
        cmaps = ee.Image.cat(cmap,smap,fmap,bmap).rename(['cmap','smap','fmap']+timestamplist1[1:])  
        assexport = ee.batch.Export.image.toAsset(cmaps.byte().clip(poly),
                                    description='assetExportTask', 
                                    assetId=w_exportassetsname.value,scale=10,maxPixels=1e9)      
        assexport.start()
        with w_out: 
            w_out.clear_output() 
            print('Exporting change maps to %s\n task id: %s'%(w_exportassetsname.value,str(assexport.id)))
    except Exception as e:
        with w_out:
            print('Error: %s'%e)                                          
    
w_export_ass.on_click(on_export_ass_button_clicked)  

def on_export_drv_button_clicked(b):
    ''' Export to Google Drive
    '''
    try:
        smap = ee.Image(result.get('smap')).byte()
        cmap = ee.Image(result.get('cmap')).byte()
        fmap = ee.Image(result.get('fmap')).byte() 
        bmap = ee.Image(result.get('bmap')).byte()            
        cmaps = ee.Image.cat(cmap,smap,fmap,bmap).rename(['cmap','smap','fmap']+timestamplist1[1:])  
        fileNamePrefix=w_exportdrivename.value.replace('/','-')            
        gdexport = ee.batch.Export.image.toDrive(cmaps.byte().clip(poly),
                                    description='driveExportTask', 
                                    folder = 'gee',
                                    fileNamePrefix=fileNamePrefix,scale=10,maxPixels=1e9)   
        gdexport.start()
        with w_out:
            w_out.clear_output()
            print('Exporting change maps to Drive/gee/%s\n task id: %s'%(fileNamePrefix,str(gdexport.id))) 
    except Exception as e:
        with w_out:
            print('Error: %s'%e) 

w_export_drv.on_click(on_export_drv_button_clicked)            

def on_plot_button_clicked(b):          
#  plot change fractions        
    global bmap1 
    def plot_iter(current,prev):
        current = ee.Image.constant(current)
        plots = ee.List(prev) 
        res = bmap1.multiply(0) \
                  .where(bmap1.eq(current),1) \
                  .reduceRegion(ee.Reducer.mean(),scale=10,maxPixels=10e10)
        return ee.List(plots.add(res))
    with w_out:
        try:
            w_out.clear_output()            
            print('Change fraction plots ...')                  
            assetImage = ee.Image(w_exportassetsname.value)
            k = assetImage.bandNames().length().subtract(4).getInfo()            
            bmap1 = assetImage.select(ee.List.sequence(3,k+2))            
            if w_maskwater.value:
                bmap1 = bmap1.updateMask(watermask) 
            plots = ee.List(ee.List([1,2,3]).iterate(plot_iter,ee.List([]))).getInfo()           
            bns = np.array(list([s[3:9] for s in list(plots[0].keys())])) 
            x = range(1,k+1)  
            _ = plt.figure(figsize=(10,5))
            plt.plot(x,list(plots[0].values()),'ro-',label='posdef')
            plt.plot(x,list(plots[1].values()),'co-',label='negdef')
            plt.plot(x,list(plots[2].values()),'yo-',label='indef')        
            ticks = range(0,k+2)
            labels = [str(i) for i in range(0,k+2)]
            labels[0] = ' '
            labels[-1] = ' '
            labels[1:-1] = bns 
            if k>50:
                for i in range(1,k+1,2):
                    labels[i] = ''
            plt.xticks(ticks,labels,rotation=90)
            plt.legend()
            fn = w_exportassetsname.value.replace('/','-')+'.png'
            plt.savefig(fn,bbox_inches='tight') 
            w_out.clear_output()
            plt.show()
            print('Saved to ~/%s'%fn)
        except Exception as e:
            print('Error: %s'%e)               
    
w_plot.on_click(on_plot_button_clicked)


#@title Run the interface
def run():
    global m, center
    center = [51.0,6.4]
    osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    ews = basemap_to_tiles(basemaps.Esri.WorldStreetMap)
    ewi = basemap_to_tiles(basemaps.Esri.WorldImagery)
    
    dc = DrawControl(polyline={},circlemarker={})
    dc.rectangle = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.05}}
    dc.polygon = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.05}}

    dc.on_draw(handle_draw)
    
    lc = LayersControl(position='topright')
 
    m = Map(center=center, 
                    zoom=11, 
                    layout={'height':'600px','width':'1000px'},
                    layers=(ewi,ews,osm),
                    controls=(dc,lc))
    with w_out:
        w_out.clear_output()
        print('Algorithm output') 
    display(m) 
    return box      