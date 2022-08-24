import pandas as pd
from functools import reduce
import numpy as np
class dataset:
    def read_csv(csv_path):
        df = pd.read_csv(csv_path)
        df_renamed = df.rename({'Reading Time (CST)':'Date'}, axis=1)
        if '0 to 5 cm Depth Average WFV (%)' in df_renamed: 
            df_no_data_removed = df_renamed.loc[df_renamed['0 to 5 cm Depth Average WFV (%)'] != 'No Data']
            df_no_data_removed['0 to 5 cm Depth Average WFV (%)'] = pd.to_numeric(df_no_data_removed['0 to 5 cm Depth Average WFV (%)'], 
                                                                                  downcast="float",
                                                                                  errors ='coerce')
            df_no_data_removed =  df_no_data_removed.rename({'0 to 5 cm Depth Average WFV (%)':'0-5cm Soil'},axis=1)

        if '0-5 cm Depth Average WFV (%)' in df_renamed:
            df_no_data_removed = df_renamed.loc[df_renamed['0-5 cm Depth Average WFV (%)'] != 'No Data']
            df_no_data_removed['0-5 cm Depth Average WFV (%)'] = pd.to_numeric(df_no_data_removed['0-5 cm Depth Average WFV (%)'],
                                                                               downcast="float",
                                                                               errors='coerce')
            df_no_data_removed =  df_no_data_removed.rename({'0-5 cm Depth Average WFV (%)':'0-5cm Soil'},axis=1)

        df_no_data_removed['Date'] =  pd.to_datetime(df_no_data_removed['Date'], 
                                                     infer_datetime_format=True, 
                                                     format='%d/%m/%y')
        return df_no_data_removed   
    
    def get_soil_prep_date(csv_path,prename):
        '''
        returns soil and prep data list of 13 stations.
        usage:
        
        csv_path = "/Users/alperbalmumcu/Github/sen1-sen2-soil-moisture/RISMA/dataset/datasets/"
        prename = "Manitoba_Station_""
        soil, prep = get_soil_prep(csv_path,prename)
        '''
        soil_list = []
        prep_list = []
        date_list = []
        
        for idx in range(1,14):
            read_csv = dataset.read_csv(csv_path + prename + str(idx) + ".csv")
            soil_list.append(read_csv['0-5cm Soil'])
            prep_list.append(read_csv['Precipitation Precip (mm)'])
            date_list.append(read_csv['Date'])
            
        return soil_list,prep_list,date_list
    
    def merge_datasets_wdates(date_list,soil_list,prep_list):
        '''
        inputs are date_list, soil_list and prep_list respectfully.
        returns merged dataframe.
        '''
        df_list = []
        for i in range(13):
            dfs = pd.DataFrame(list(zip(date_list[i],soil_list[i],prep_list[i])), 
                                columns= ['Date', f'Soil_{str(i+1)}',f'Prep_{str(i+1)}'])
            df_list.append(dfs)
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                                    how='outer'), df_list)
        df = df_merged.replace(np.nan,'',regex=True)
        df = df.replace('No Data', '')

        return df
    
    
    
    

        

                    

