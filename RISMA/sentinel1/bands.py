import pandas as pd
import ee, datetime
import numpy as np

def Sentinel1_extraction(_area_list):
  station13_bf = []

  s1_bands_features_names = ['Date','VV','VH','VH/VV','q_ratio',
                             'Mc','Theta','Entropy']

  df_sentinel_1_list = []

  station = 1
  for Area in _area_list: 

    def VV_band(img):
      temp = img.select('VV')
      temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
      return img.addBands(temp).set(temp_img)

    def VH_band(img):
      temp = img.select('VH')
      temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
      return img.addBands(temp).set(temp_img)

    def VH_VV(img):
      vh = img.select('VH')
      vv = img.select('VV')
      vh_over_vv = vh.divide(vv).rename('VH/VV')
      mean = vh_over_vv.reduceRegion(ee.Reducer.mean(),Area)
      return img.addBands(vh_over_vv).set(mean)

    def q_ratio(img):
      vh = img.select('VH')
      vv = img.select('VV')
      vh_over_vv = vh.divide(vv).rename('q_ratio')
      mean = vh_over_vv.reduceRegion(ee.Reducer.mean(),Area)
      return img.addBands(vh_over_vv).set(mean)


    def purity(q_list):
      mc = []
      for i in q_list:
        temp = (1-i)/(1+i)
        mc.append(temp)
      return mc

    def theta(q_list):
      theta = []
      for i in q_list:
        temp = ((1-i)**2)/(1-i+i**2)
        temp = np.arctan(temp)
        theta.append(temp)
      return theta

    def entropy(q_list):
      entropy = []
      for i in q_list:
        p1 = 1/(1+i)
        p2 = i/(1+i)
        sum = (p1*np.log2(p1) + p2*np.log2(p2)) * -1
        entropy.append(sum)
      return entropy

    collection = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                  .filterDate('2018-11-30','2021-10-08')
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .select(['VV','VH'])
                  # .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                  .filterBounds(Area)
                  .filter(ee.Filter.eq('resolution_meters', 10))
                  .filter(ee.Filter.eq('resolution', 'H')))
    

    count = collection.size().getInfo()
    print(f'Sentinel 1 Station {station} Collection Size: ', count)


    _date= ee.List(collection.map(VV_band).select('VV').aggregate_array('system:time_start')).getInfo()
    s1_dates = [datetime.datetime.fromtimestamp(x // 1000).strftime('%Y/%m/%d') for x in _date]


    bands_features_names = ['VV','VH','VH/VV','q_ratio']
    
    bands_features_funcs = [VV_band,VH_band,VH_VV,q_ratio]

    collection_list = []
    for idx in range(3):
      if idx == 0:
        collection_list.append(s1_dates)
        
      collection_list.append(ee.List(collection.map(bands_features_funcs[idx]) \
                                    .select(bands_features_names[idx]) \
                                    .aggregate_array(bands_features_names[idx])) \
                                    .getInfo())

      if idx == 2:
        q_list  = ee.List(collection.map(bands_features_funcs[idx+1]) \
                                    .select(bands_features_names[idx+1]) \
                                    .aggregate_array(bands_features_names[idx+1])) \
                                    .getInfo()
        collection_list.append(q_list)
        collection_list.append(purity(q_list))
        collection_list.append(theta(q_list))
        collection_list.append(entropy(q_list))
              
    
    station13_bf.append(collection_list)
    station += 1
  

  for i in range(13):
    dfs = pd.DataFrame(list(zip(pd.to_datetime(station13_bf[i][0]).strftime('%Y/%m/%d'),
                                station13_bf[i][1],station13_bf[i][2],station13_bf[i][3],
                                station13_bf[i][4],station13_bf[i][5],station13_bf[i][6],
                                station13_bf[i][7])), 
                        columns = s1_bands_features_names)
    df_sentinel_1_list.append(dfs)

  return df_sentinel_1_list
