import ee

def maskS2clouds(image):
  cloudBitMask = 1024
  cirrusBitMask = 2048
  qa = image.select('QA60')
  mask = qa.bitwiseAnd(cloudBitMask).eq(0) and qa.bitwiseAnd(cirrusBitMask).eq(0)
  return image.updateMask(mask)


def MSI(img):
  nir = img.select('B8')
  swir1 = img.select('B11')
  msi = swir1.divide(nir).rename('MSI')
  meanMSI = msi.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(msi).set(meanMSI)


def MSI2(img):
  nir = img.select('B8')
  swir2 = img.select('B12')
  msi2 = swir2.divide(nir).rename('MSI_2')
  meanMSI2 = msi2.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(msi2).set(meanMSI2)


def NDMI(img):
  ndmiImage = img.normalizedDifference(['B8','B11']).rename('NDMI')
  meanNDMI = ndmiImage.reduceRegion(ee.Reducer.mean(), Area)

  return img.addBands(ndmiImage).set(meanNDMI)

def SRWI(img):
  nir = img.select('B8')
  red = img.select('B4')
  srwi = nir.subtract(red).rename('SRWI')
  meanSRWI = srwi.reduceRegion(ee.Reducer.mean(), Area)

  return img.addBands(srwi).set(meanSRWI)

def ndvi(img):
  ndviImage = img.normalizedDifference(['B8','B4'])
  meanNDVI = ndviImage.reduceRegion(ee.Reducer.mean(), Area)

  return img.addBands(ndviImage).set(meanNDVI)

def NMDI(img):
  nir = img.select('B8')
  swir1 = img.select('B11')
  swir2 = img.select('B12')
  nmdi = nir.subtract(swir1).add(swir2).divide(nir.add(swir1).subtract(swir2)).rename('NMDI')
  meanNMDI = nmdi.reduceRegion(ee.Reducer.mean(),Area)
  
  return img.addBands(nmdi).set(meanNMDI)

def B2(img):
  temp = img.select('B2')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B3(img):
  temp = img.select('B3')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B4(img):
  temp = img.select('B4')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B5(img):
  temp = img.select('B5')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B6(img):
  temp = img.select('B6')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B7(img):
  temp = img.select('B7')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)
def B8(img):
  temp = img.select('B8')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B8A(img):
  temp = img.select('B8A')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B11(img):
  temp = img.select('B11')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)

def B12(img):
  temp = img.select('B12')
  temp_img = temp.reduceRegion(ee.Reducer.mean(),Area)
  return img.addBands(temp).set(temp_img)
