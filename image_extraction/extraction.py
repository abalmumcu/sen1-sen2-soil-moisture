import ee
from utils import center_point,Sentinel1_extract,Sentinel2_extract
ee.Authenticate()
ee.Initialize()

start_date = '2018-11-29'
end_date = '2021-10-09'

Area_1 = ee.Geometry.Point(center_point(-98.02339978439299,49.56252442309637,-98.01765985710112,49.564410252567015)).buffer(300)
Area_2 = ee.Geometry.Point(center_point(-97.93329566842583,49.48817155967793,-97.92312473184136,49.49366322214269)).buffer(300)
Area_3 = ee.Geometry.Point(center_point(-97.95644712928592,49.517764130065665,-97.95112562660037,49.5205432025645)).buffer(300)
Area_4 = ee.Geometry.Point(center_point(-97.99377253479413,49.636173539763185,-97.98428824371747,49.63998112090547)).buffer(300)
Area_5 = ee.Geometry.Point(center_point(-97.95926229089906,49.62142930253476,-97.95176283449342,49.62356653928279)).buffer(300)
Area_6 = ee.Geometry.Point(center_point(-97.95923407290029,49.674205617579375,-97.949277713037,49.67939869275389)).buffer(300)
Area_7 = ee.Geometry.Point(center_point(-98.01040614461421,49.66569342866649,-98.00595904206752,49.66823841127352)).buffer(300)
Area_8 = ee.Geometry.Point(center_point(-97.98171534643836,49.75028711801634,-97.9722310553617,49.7531985198659)).buffer(300)
Area_9 = ee.Geometry.Point(center_point(-98.02717792198182,49.69154411877749,-98.0177901904297,49.69438964418434)).buffer(300)
Area_10 = ee.Geometry.Point(center_point(-97.34998515609742,49.97399539493168,-97.34472802642823,49.97528569490996)).buffer(300)
Area_11 = ee.Geometry.Point(center_point(-97.57354462917826,50.10901804255274,-97.56657088573954,50.1117564877798)).buffer(300)
Area_12 = ee.Geometry.Point(center_point(-97.59943425298691,50.189721550412614,-97.59752720237732,50.19036037342664)).buffer(300)
Area_13 = ee.Geometry.Point(center_point(-99.388888036558,49.93140549249452,-99.38559428388771,49.932869584113625)).buffer(300)

AreaList = [Area_1,Area_2,Area_3,Area_4,Area_5,Area_6,Area_7,Area_8,Area_9,Area_10,Area_11,Area_12,Area_13]

# for station,area in enumerate(AreaList,1):
#     # Sentinel1_extract(start_date,end_date,area,station)
#     Sentinel2_extract(start_date,end_date,area,station)

if __name__ == '__main__':
    Sentinel2_extract(start_date,end_date,Area_1,'1')
