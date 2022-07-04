import ee
import numpy as np

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
