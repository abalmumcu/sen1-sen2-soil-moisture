import pandas as pd
import ee
import datetime
import numpy as np
from itertools import count as counter

ee.Authenticate()
ee.Initialize()

class risma_gee_band_indice_extracter:
    def __init__(self,dataset_list, gee_area_longtitude = [-98.01924,-97.93374], gee_area_latitude = [49.56234,49.49250], gee_buffer = [10,4]):
        self.datasets = dataset_list
        self.df_modified_risma_list = []
        self.avaliable_dates_list = []
        self.avaliable_data_list = []
        self.avaliable_prep_list = []
        self.gee_buffer = gee_buffer
        self.gee_area_longtitude = gee_area_longtitude
        self.gee_area_latitude = gee_area_latitude
        self.gee_date_start = '2018-11-29'
        self.gee_date_end = '2023-01-01'

        self.s2_dates = []
        self.s1_dates = []

        # Sentinel2 Indices
        self.s2_ndvi_list =[]
        self.s2_nmdi_list = []
        self.s2_ndmi_list = []
        self.s2_srwi_list = []
        self.s2_msi_list = []
        self.s2_msi2_list = []
    
        # Sentinel2 Bands
        self.s2_b2_list =[]
        self.s2_b3_list =[]
        self.s2_b4_list =[]
        self.s2_b5_list =[]
        self.s2_b6_list =[]
        self.s2_b7_list =[]
        self.s2_b8_list =[]
        self.s2_b8a_list =[]
        self.s2_b11_list =[]
        self.s2_b12_list =[]

        # Sentinel1 Polarization Bands and Indices
        self.s1_vv_list = []
        self.s1_vh_list = []
        self.s1_vh_vv_list = []
        self.s1_theta_list = []
        self.s1_mc_list = []
        self.s1_entropy_list = []

    def risma_soil_dataset(self):
        for station_idx,csv_dataset,area_long,area_latitude,buff in zip(counter(), self.datasets,self.gee_area_longtitude,self.gee_area_latitude,self.gee_buffer):
            print(f"Process starting for station{station_idx+1}")
            df = pd.read_csv(csv_dataset)
            df_renamed = df.rename({'Reading Time (CST)':'Date'}, axis=1)
            df_no_data_removed = df_renamed.replace('No Data', '0')
            df_no_data_removed["0 to 5 cm Depth Average WFV (%)"] = pd.to_numeric(df_no_data_removed["0 to 5 cm Depth Average WFV (%)"], downcast="float")
            df_no_data_removed = df_no_data_removed.drop(df_no_data_removed[df_no_data_removed["0 to 5 cm Depth Average WFV (%)"] == 0].index)
            df_clean_risma = df_no_data_removed.copy()
            df_no_data_removed['Date'] =  pd.to_datetime(df_no_data_removed['Date'], infer_datetime_format=True)
            self.df_modified_risma_list.append(df_no_data_removed)

            no_data_dates = []
            avaliable_data_dates = []
            avaliable_datas = []
            avaliable_prep_data = []

            dates = []
            soil_datas = []
            prep_data = []

            dates = df_clean_risma["Date"].tolist()
            soil_datas = df_clean_risma["0 to 5 cm Depth Average WFV (%)"].tolist()
            prep_data = df_clean_risma['Precipitation Precip (mm)'].tolist()

            for date,soil,prep in zip(dates,soil_datas,prep_data):
                if soil == 'No Data':
                    no_data_dates.append(date)
                else:
                    avaliable_data_dates.append(date)
                    avaliable_datas.append(soil)
                    avaliable_prep_data.append(prep)

            self.avaliable_dates_list.append(avaliable_data_dates)
            self.avaliable_data_list.append(avaliable_datas)
            self.avaliable_prep_list.append(avaliable_prep_data)

            Area = ee.Geometry.Point(area_long,area_latitude).buffer(buff)

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


            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterDate(self.gee_date_start,self.gee_date_end)
                    .map(maskS2clouds)
                    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 1)
                    .select(['B2','B3','B4','B5','B6','B7','B8','B8A','QA60','B11','B12'])
                    .filterBounds(Area))

            
            count = s2_collection.size().getInfo()
            print('Sentinel2 Total Images:', count)


            date__= ee.List(s2_collection.map(ndvi).select('nd').aggregate_array('system:time_start')).getInfo()
            self.s2_dates.append([datetime.datetime.fromtimestamp(x // 1000).strftime('%Y-%m-%d') for x in date__])

            self.s2_ndvi_list.append(ee.List(s2_collection.map(ndvi).select('nd').aggregate_array('nd')).getInfo())
            self.s2_ndmi_list.append(ee.List(s2_collection.map(NDMI).select('NDMI').aggregate_array('NDMI')).getInfo())
            self.s2_srwi_list.append(ee.List(s2_collection.map(SRWI).select('SRWI').aggregate_array('SRWI')).getInfo())
            self.s2_nmdi_list.append(ee.List(s2_collection.map(NMDI).select('NMDI').aggregate_array('NMDI')).getInfo())
            self.s2_msi_list.append(ee.List(s2_collection.map(MSI).select('MSI').aggregate_array('MSI')).getInfo())
            self.s2_msi2_list.append(ee.List(s2_collection.map(MSI2).select('MSI_2').aggregate_array('MSI_2')).getInfo())
            self.s2_b2_list.append(ee.List(s2_collection.map(B2).select('B2').aggregate_array('B2')).getInfo())
            self.s2_b3_list.append(ee.List(s2_collection.map(B3).select('B3').aggregate_array('B3')).getInfo())
            self.s2_b4_list.append(ee.List(s2_collection.map(B4).select('B4').aggregate_array('B4')).getInfo())
            self.s2_b5_list.append(ee.List(s2_collection.map(B5).select('B5').aggregate_array('B5')).getInfo())
            self.s2_b6_list.append(ee.List(s2_collection.map(B6).select('B6').aggregate_array('B6')).getInfo())
            self.s2_b7_list.append(ee.List(s2_collection.map(B7).select('B7').aggregate_array('B7')).getInfo())
            self.s2_b8_list.append(ee.List(s2_collection.map(B8).select('B8').aggregate_array('B8')).getInfo())
            self.s2_b8a_list.append(ee.List(s2_collection.map(B8A).select('B8A').aggregate_array('B8A')).getInfo())
            self.s2_b11_list.append(ee.List(s2_collection.map(B11).select('B11').aggregate_array('B11')).getInfo())
            self.s2_b12_list.append(ee.List(s2_collection.map(B12).select('B12').aggregate_array('B12')).getInfo())


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

            s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                    .filterDate(self.gee_date_start,self.gee_date_end)
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .select(['VV','VH'])
                    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                    .filterBounds(Area)
                    .filter(ee.Filter.eq('resolution_meters', 10))
                    .filter(ee.Filter.eq('resolution', 'H')))

            count = s1_collection.size().getInfo()
            print('Sentinel1 Total Images:', count)



            # Dates
            date_SAR= ee.List(s1_collection.map(VV_band).select('VV').aggregate_array('system:time_start')).getInfo()
            self.s1_dates.append([datetime.datetime.fromtimestamp(x // 1000).strftime('%Y-%m-%d') for x in date_SAR])

            self.s1_vv_list.append(ee.List(s1_collection.map(VV_band).select('VV').aggregate_array('VV')).getInfo())
            self.s1_vh_list.append(ee.List(s1_collection.map(VH_band).select('VH').aggregate_array('VH')).getInfo())
            self.s1_vh_vv_list.append(ee.List(s1_collection.map(VH_VV).select('VH/VV').aggregate_array('VH/VV')).getInfo())
            #q_list = ee.List(q_ratio_col = s1_collection.map(q_ratio).select('q_ratio').aggregate_array('q_ratio')).getInfo()
            self.s1_mc_list.append(purity(ee.List(s1_collection.map(q_ratio).select('q_ratio').aggregate_array('q_ratio')).getInfo()))
            self.s1_theta_list.append(theta(ee.List(s1_collection.map(q_ratio).select('q_ratio').aggregate_array('q_ratio')).getInfo()))
            self.s1_entropy_list.append(entropy(ee.List(s1_collection.map(q_ratio).select('q_ratio').aggregate_array('q_ratio')).getInfo()))

            print(f"Process complated for station{station_idx+1}")

        return self.avaliable_data_list, self.avaliable_dates_list, self.avaliable_prep_list, self.s2_dates, self.s2_srwi_list,self.s2_ndvi_list, \
                self.s2_ndmi_list, self.s2_nmdi_list, self.s2_msi_list, self.s2_msi2_list, \
                self.s2_b2_list,self.s2_b3_list, self.s2_b4_list, self.s2_b5_list, self.s2_b6_list, \
                self.s2_b7_list,self.s2_b8_list, self.s2_b8a_list, self.s2_b11_list,self.s2_b12_list, \
                self.s1_dates,self.s1_vv_list,self.s1_vh_list,self.s1_vh_vv_list,self.s1_mc_list,self.s1_theta_list,self.s1_entropy_list



class datasets:
    def __init__(self, avaliable_datas, avaliable_data_dates, avaliable_prep_data, s2_dates, srwi_list,ndvi_list,ndmi_list,nmdi_list,msi_list,msi_2_list, 
                b2_list,b3_list, b4_list, b5_list, b6_list, b7_list,b8_list, b8A_list, b11_list,b12_list,
                s1_dates,vv_list,vh_list,vh_vv_list,mc_list,theta_list,entropy_list):

        # Risma data
        self.avaliable_datas =avaliable_datas
        self.avaliable_data_dates =avaliable_data_dates
        self.avaliable_prep_data = avaliable_prep_data        
        
        # Sentinel 1 datas
        self.s1_dates =s1_dates 
        self.vv_list = vv_list
        self.vh_list = vh_list
        self.vh_vv_list = vh_vv_list
        self.mc_list =mc_list
        self.theta_list = theta_list
        self.entropy_list = entropy_list       

        # Sentinel 2 datas
        self.s2_dates =s2_dates
        self.ndvi_list = ndvi_list
        self.nmdi_list =nmdi_list
        self.ndmi_list =ndmi_list
        self.msi_list = msi_list
        self.msi_2_list = msi_2_list
        self.srwi_list = srwi_list
        self.b2_list = b2_list
        self.b3_list = b3_list
        self.b4_list = b4_list
        self.b5_list = b5_list
        self.b6_list = b6_list
        self.b7_list = b7_list
        self.b8_list = b8_list
        self.b8A_list = b8A_list
        self.b11_list = b11_list
        self.b12_list = b12_list

        # Sentinel 1 match data
        self.s1_match_dates = []
        self.s1_match_dataset_soil = []
        self.s1_match_prep = []
        self.s1_match_vv = []
        self.s1_match_vh = []
        self.s1_match_vh_vv = []
        self.s1_match_mc = []
        self.s1_match_theta = []
        self.s1_match_entropy = []
        
        # Sentinel 2 match data
        self.one_match_dates = []
        self.one_match_dataset_soil = []
        self.one_match_ndvi = []
        self.one_match_srwi = []
        self.one_match_ndmi = []
        self.one_match_nmdi = []
        self.one_match_msi = []
        self.one_match_msi_2 = []
        self.one_match_prep = []

        self.second_match_dates = []
        self.second_match_dataset_soil = []
        self.second_match_ndvi = []
        self.second_match_srwi = []
        self.second_match_ndmi = []
        self.second_match_nmdi = []
        self.second_match_msi = []
        self.second_match_msi_2 = []
        self.second_match_prep = []
        self.second_match_b2 = []
        self.second_match_b3 = []
        self.second_match_b4 = []
        self.second_match_b5 = []
        self.second_match_b6 = []
        self.second_match_b7 = []
        self.second_match_b8 = []
        self.second_match_b8A = []
        self.second_match_b11 = []
        self.second_match_b12 = []

        # Sentinel1 and Sentinel2 match data 
        self.match_dates = []
        self.match_dataset_soil = []
        self.match_ndvi = []
        self.match_srwi = []
        self.match_ndmi = []
        self.match_nmdi = []
        self.match_msi = []
        self.match_msi_2 = []
        self.match_prep = []
        self.match_b2 = []
        self.match_b3 = []
        self.match_b4 = []
        self.match_b5 = []
        self.match_b6 = []
        self.match_b7 = []
        self.match_b8 = []
        self.match_b8A = []
        self.match_b11 = []
        self.match_b12 = []
        self.match_vv = []
        self.match_vh = []
        self.match_vh_vv = []
        self.match_mc=[]
        self.match_theta=[]
        self.match_entropy=[]

        self.Sentinel2_indices()
        self.Sentinel2_indices_bands()
        self.Sentinel2_Sentinel1_match()
        self.Sentinel1_match()

    def Sentinel2_indices(self):
        for dataset_soil,dataset_date,pre in zip(self.avaliable_datas,self.avaliable_data_dates,self.avaliable_prep_data):
            for gee_ndvi,gee_date,srwi,ndmi,nmdi,msi,msi2 in zip(self.ndvi_list,self.s2_dates,self.srwi_list,self.ndmi_list,self.nmdi_list,self.msi_list,self.msi_2_list):
                if gee_date == dataset_date : 
                    self.one_match_dates.append(gee_date)
                    self.one_match_ndvi.append(gee_ndvi)
                    self.one_match_dataset_soil.append(dataset_soil)
                    self.one_match_srwi.append(srwi)
                    self.one_match_ndmi.append(ndmi)
                    self.one_match_nmdi.append(nmdi)
                    self.one_match_msi.append(msi)
                    self.one_match_msi_2.append(msi2)
                    self.one_match_prep.append(pre)


    def Sentinel2_indices_bands(self):
        for dataset_soil,dataset_date,pre in zip(self.avaliable_datas,self.avaliable_data_dates,self.avaliable_prep_data):
            for gee_ndvi,gee_date,srwi,ndmi,nmdi,msi,msi2,b2,b3,b4,b5,b6,b7,b8,b8a,b11,b12 in zip(self.ndvi_list,
                    self.s2_dates,self.srwi_list,self.ndmi_list,self.nmdi_list,self.msi_list,self.msi_2_list,self.b2_list,self.b3_list,self.b4_list,
                    self.b5_list,self.b6_list,self.b7_list,self.b8_list,self.b8A_list,self.b11_list,self.b12_list):

                if gee_date == dataset_date : 
                    self.second_match_dates.append(gee_date)
                    self.second_match_ndvi.append(gee_ndvi)
                    self.second_match_dataset_soil.append(dataset_soil)
                    self.second_match_srwi.append(srwi)
                    self.second_match_ndmi.append(ndmi)
                    self.second_match_nmdi.append(nmdi)
                    self.second_match_msi.append(msi)
                    self.second_match_msi_2.append(msi2)
                    self.second_match_b2.append(b2)
                    self.second_match_b3.append(b3)
                    self.second_match_b4.append(b4)
                    self.second_match_b5.append(b5)
                    self.second_match_b6.append(b6)
                    self.second_match_b7.append(b7)
                    self.second_match_b8.append(b8)
                    self.second_match_b8A.append(b8a)
                    self.second_match_b11.append(b11)
                    self.second_match_b12.append(b12)
                    self.second_match_prep.append(pre)
                

    def Sentinel2_Sentinel1_match(self):
        for dataset_soil,dataset_date,pre in zip(self.avaliable_datas,self.avaliable_data_dates,self.avaliable_prep_data):
            for gee_ndvi,gee_date,srwi,ndmi,nmdi,msi,msi2,b2,b3,b4,b5,b6,b7,b8,b8a,b11,b12 in zip(self.ndvi_list,
                    self.s2_dates,self.srwi_list,self.ndmi_list,self.nmdi_list,self.msi_list,self.msi_2_list,self.b2_list,self.b3_list,self.b4_list,
                    self.b5_list,self.b6_list,self.b7_list,self.b8_list,self.b8A_list,self.b11_list,self.b12_list):
                for s1_date,vv,vh,vh_vv,mc,theta,entr in zip(self.s1_dates,self.vv_list,self.vh_list,self.vh_vv_list,self.mc_list,self.theta_list,self.entropy_list):

                    if gee_date == dataset_date == s1_date: 
                        self.match_dates.append(gee_date)
                        self.match_ndvi.append(gee_ndvi)
                        self.match_dataset_soil.append(dataset_soil)
                        self.match_srwi.append(srwi)
                        self.match_ndmi.append(ndmi)
                        self.match_nmdi.append(nmdi)
                        self.match_msi.append(msi)
                        self.match_msi_2.append(msi2)
                        self.match_b2.append(b2)
                        self.match_b3.append(b3)
                        self.match_b4.append(b4)
                        self.match_b5.append(b5)
                        self.match_b6.append(b6)
                        self.match_b7.append(b7)
                        self.match_b8.append(b8)
                        self.match_b8A.append(b8a)
                        self.match_b11.append(b11)
                        self.match_b12.append(b12)     
                        self.match_prep.append(pre)
                        self.match_vv.append(vv)
                        self.match_vh.append(vh)
                        self.match_vh_vv.append(vh_vv)
                        self.match_mc.append(mc)
                        self.match_theta.append(theta)
                        self.match_entropy.append(entr)


    def Sentinel1_match(self):
        for dataset_soil,dataset_date,pre in zip(self.avaliable_datas,self.avaliable_data_dates,self.avaliable_prep_data):
            for s1_date,vv,vh,vh_vv,mc,theta,entr in zip(self.s1_dates,self.vv_list,self.vh_list,self.vh_vv_list,self.mc_list,self.theta_list,self.entropy_list):

                if s1_date == dataset_date : 
                    self.s1_match_dataset_soil.append(dataset_soil)
                    self.s1_match_prep.append(pre)
                    self.s1_match_dates.append(s1_date)
                    self.s1_match_vv.append(vv)
                    self.s1_match_vh.append(vh)
                    self.s1_match_vh_vv.append(vh_vv)
                    self.s1_match_mc.append(mc)
                    self.s1_match_theta.append(theta)
                    self.s1_match_entropy.append(entr)

    
    def dataset_sentinel1(self,precipitation=False,save=False,save_path=''):
        if precipitation == True:
            df_S1 = pd.DataFrame(list(zip(self.s1_match_dates, self.s1_match_dataset_soil,self.s1_match_prep,self.s1_match_vv,
                                        self.s1_match_vh,self.s1_match_vh_vv,self.s1_match_mc,self.s1_match_theta,self.s1_match_entropy)),
               columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','Precipitation','VV','VH','VH/VV','Mc','Theta','Entropy'])
        else:
            df_S1 = pd.DataFrame(list(zip(self.s1_match_dates, self.s1_match_dataset_soil,self.s1_match_vv,
                                        self.s1_match_vh,self.s1_match_vh_vv,self.s1_match_mc,self.s1_match_theta,self.s1_match_entropy)),
               columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','VV','VH','VH/VV','Mc','Theta','Entropy'])
        df_S1["RISMA 0 to 5 cm Depth Average WFV (%)"] = pd.to_numeric(df_S1["RISMA 0 to 5 cm Depth Average WFV (%)"], downcast="float")
        df_S1['Date'] =  pd.to_datetime(df_S1['Date'], infer_datetime_format=True)

        if save == True:
            df_S1.to_csv(save_path)

        df_S1.loc[:, df_S1.columns != 'Date'].apply(pd.to_numeric)
        df_S1['Date'] =  pd.to_datetime(df_S1['Date'], infer_datetime_format=True)
        return df_S1
    

    def dataset_sentinel2_indices(self,precipitation=False,save=False,save_path=''):
        if precipitation == True:
            df = pd.DataFrame(list(zip(self.one_match_dates, self.one_match_dataset_soil, self.one_match_prep ,self.one_match_msi,
                            self.one_match_msi_2, self.one_match_ndvi, self.one_match_srwi, self.one_match_ndmi,self.one_match_nmdi)),
               columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','Precipitation','MSI','MSI_2','NDVI','SRWI','NDMI','NMDI'])
        else:
            df = pd.DataFrame(list(zip(self.one_match_dates, self.one_match_dataset_soil,self.one_match_msi,
                            self.one_match_msi_2, self.one_match_ndvi, self.one_match_srwi, self.one_match_ndmi,self.one_match_nmdi)),
               columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','MSI','MSI_2','NDVI','SRWI','NDMI','NMDI'])
        if save == True:
            df.to_csv(save_path)
        df["RISMA 0 to 5 cm Depth Average WFV (%)"] = pd.to_numeric(df["RISMA 0 to 5 cm Depth Average WFV (%)"], downcast="float")
        df['NDVI'] = pd.to_numeric(df["NDVI"], downcast="float")
        df['MSI'] = pd.to_numeric(df["MSI"], downcast="float")
        df['MSI_2'] = pd.to_numeric(df["MSI_2"], downcast="float")
        df['SRWI'] = pd.to_numeric(df["SRWI"], downcast="float")
        df['NDMI'] = pd.to_numeric(df["NDMI"], downcast="float")
        df['NMDI'] = pd.to_numeric(df["NMDI"], downcast="float")

        df['Date'] =  pd.to_datetime(df['Date'], infer_datetime_format=True)
        return df

    def dataset_sentinel2_indices_bands(self,precipitation=False,save=False,save_path=''):
        if precipitation == True:
            df_bands = pd.DataFrame(list(zip(self.second_match_dates, self.second_match_dataset_soil,self.second_match_prep,
                    self.second_match_msi,self.second_match_msi_2, self.second_match_ndvi,self.second_match_srwi,
                    self.second_match_ndmi,self.second_match_nmdi,self.second_match_b2,self.second_match_b3,self.second_match_b4,
                    self.second_match_b5,self.second_match_b6,self.second_match_b7,self.second_match_b8,self.second_match_b8A,
                    self.second_match_b11,self.second_match_b12)),
               columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','Precipitation','MSI','MSI_2','NDVI','SRWI',
                        'NDMI','NMDI','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
        else:
            df_bands = pd.DataFrame(list(zip(self.second_match_dates, self.second_match_dataset_soil,
                        self.second_match_msi,self.second_match_msi_2, self.second_match_ndvi,self.second_match_srwi,
                        self.second_match_ndmi,self.second_match_nmdi,self.second_match_b2,self.second_match_b3,self.second_match_b4,
                        self.second_match_b5,self.second_match_b6,self.second_match_b7,self.second_match_b8,self.second_match_b8A,
                        self.second_match_b11,self.second_match_b12)),
                columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','MSI','MSI_2','NDVI','SRWI',
                            'NDMI','NMDI','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
        if save == True:
            df_bands.to_csv(save_path)

        df_bands.loc[:, df_bands.columns != 'Date'].apply(pd.to_numeric)
        df_bands['Date'] =  pd.to_datetime(df_bands['Date'], infer_datetime_format=True)
        return df_bands
    
    def dataset_sentinel2_sentinel1_match(self,precipitation=False, save = False, save_path=''):
        if precipitation == True:
            df_S1_S2 = pd.DataFrame(list(zip(self.match_dates, self.match_dataset_soil,self.match_prep,self.match_msi,self.match_msi_2,
                                            self.match_ndvi,self.match_srwi,self.match_ndmi,self.match_nmdi,self.match_b4,
                                            self.match_b5,self.match_b6,self.match_b7,self.match_b8,self.match_b8A,self.match_b11,
                                            self.match_b12,self.match_vv,self.match_vh,self.match_vh_vv,self.match_mc,
                                            self.match_theta,self.match_entropy)),
                columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','Precipitation','MSI','MSI_2','NDVI','SRWI','NDMI',
                            'NMDI','B4','B5','B6','B7','B8','B8A','B11','B12','VV','VH','VH/VV','Mc','Theta','Entropy'])
        else:
            df_S1_S2 = pd.DataFrame(list(zip(self.match_dates, self.match_dataset_soil,self.match_msi,self.match_msi_2,
                                            self.match_ndvi,self.match_srwi,self.match_ndmi,self.match_nmdi,self.match_b4,
                                            self.match_b5,self.match_b6,self.match_b7,self.match_b8,self.match_b8A,self.match_b11,
                                            self.match_b12,self.match_vv,self.match_vh,self.match_vh_vv,self.match_mc,
                                            self.match_theta,self.match_entropy)),
                columns =['Date', 'RISMA 0 to 5 cm Depth Average WFV (%)','MSI','MSI_2','NDVI','SRWI','NDMI',
                            'NMDI','B4','B5','B6','B7','B8','B8A','B11','B12','VV','VH','VH/VV','Mc','Theta','Entropy'])

        df_S1_S2["RISMA 0 to 5 cm Depth Average WFV (%)"] = pd.to_numeric(df_S1_S2["RISMA 0 to 5 cm Depth Average WFV (%)"], downcast="float")
        df_S1_S2['Date'] =  pd.to_datetime(df_S1_S2['Date'], infer_datetime_format=True)
        if save == True:
            df_S1_S2.to_csv(save_path)
        df_S1_S2.loc[:, df_S1_S2.columns != 'Date'].apply(pd.to_numeric)
        df_S1_S2['Date'] =  pd.to_datetime(df_S1_S2['Date'], infer_datetime_format=True)
        return df_S1_S2
