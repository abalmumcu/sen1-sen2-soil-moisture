import ee

def maskS2clouds(image):
  cloudBitMask = 1024
  cirrusBitMask = 2048
  qa = image.select('QA60')
  mask = qa.bitwiseAnd(cloudBitMask).eq(0) and qa.bitwiseAnd(cirrusBitMask).eq(0)
  return image.updateMask(mask)


def center_point(firstA,firstB,secondA,secondB):
  return (firstA + secondA)/2 , (firstB + secondB)/2


def Sentinel1_extract(start_date,end_date,Area,station_number):
  
  collection = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                    .filterDate(start_date,end_date)
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .select(['VV','VH'])
                    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                    .filterBounds(Area)
                    .filter(ee.Filter.eq('resolution_meters', 10))
                    .filter(ee.Filter.eq('resolution', 'H')))

  scale = 10                                         
  Type = 'float'                                     
  nimg = collection.toList(collection.size().getInfo()).size().getInfo()            
  maxPixels = 1e10                                   
  print('Images found: ', nimg)
  #Image loop
  print('Downloading...')
  for i in range(nimg):
      img = ee.Image(collection.toList(nimg).get(i))

      vv = img.select('VV')
      vh = img.select('VH')
      vh_over_vv = vh.divide(vv).rename('VH/VV')
      img = img.addBands(vh_over_vv)
      img = img.select(['VV','VH','VH/VV'])

      Id = img.id().getInfo()
      date = img.date().format('yyyy-MM-dd').getInfo()

      task = ee.batch.Export.image.toDrive(img.toFloat(), 
                                          description=date,
                                          folder=f'Sentinel1_crop_{station_number}',
                                          fileNamePrefix= date,
                                          region = Area,
                                          dimensions = (256,256),
                                          maxPixels = maxPixels)
      task.start()
      task_id = task.id
      print(Id, ' = ', task.status()['state'])
  print('Finished!')


def Sentinel2_extract(start_date,end_date,Area,station_number):

  collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterDate(start_date,end_date)
                    # .filterDate('2016-02-29','2019-01-01')
                    .map(maskS2clouds)
                    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 5)
                    .select(['B2','B3','B4','B5','B6','B7','B8','B8A','QA60','B11','B12'])
                    # .select(['B2','B3','B4'])
                    .filterBounds(Area))

  scale = 10                                        
  Type = 'float'                                      
  nimg = collection.toList(collection.size().getInfo()).size().getInfo()             
  maxPixels = 1e10                                   
  print('Images found: ', nimg)
  #Image loop
  print('Downloading...')
  for i in range(nimg):
      img = ee.Image(collection.toList(nimg).get(i))
      nir = img.select('B8')
      red = img.select('B4')
      ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
      img = img.addBands(ndvi)
      swir1 = img.select('B11')
      swir2 = img.select('B12')
      nmdi = nir.subtract(swir1).add(swir2).divide(nir.add(swir1).subtract(swir2)).rename('NMDI')
      img = img.addBands(nmdi)
      srwi = nir.subtract(red).rename('SRWI')
      img = img.addBands(srwi)

      img = img.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','NMDI','SRWI'])
      Id = img.id().getInfo()
      # onlyDateId = img.get('PRODUCT_ID').getInfo().split('_')[2].split('T')[0]
      date = img.date().format('yyyy-MM-dd').getInfo()
      task = ee.batch.Export.image.toDrive(img.toFloat(), 
                                          description=date,
                                          folder=f'Sentinel2_crop_{station_number}',
                                          fileNamePrefix= date,
                                          region = Area,
                                          dimensions = (256,256),
                                          maxPixels = maxPixels)
      task.start()
      task_id = task.id
      print(Id, ' = ', task.status()['state'])
  print('Finished!')
