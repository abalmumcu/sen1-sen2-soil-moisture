import pandas as pd

class dataset:
    def __init__(self,csv_file,*args,**kwargs) -> None:
        self.csv_raw = csv_file
        self.avaliable_data_dates = []
        self.avaliable_datas = []
        self.avaliable_prep_data = []
        self.dates = []
        self.soil_datas = []
        self.prep_data = []
        self.no_data_dates = []

    def read(self,*args,**kwargs):
        df = pd.read_csv(self.csv_raw)
        df = df.rename({'Reading Time (CST)':'Date'}, axis=1)
        df = df.replace('No Data', '')

        df["0 to 5 cm Depth Average WFV (%)"] = pd.to_numeric(df["0 to 5 cm Depth Average WFV (%)"], downcast="float")
        df['Date'] =  pd.to_datetime(df['Date'], infer_datetime_format=True)
        return df

    def _no_data_extraction(self,df):
        dates = df["Date"].tolist()
        soil_datas = df["0 to 5 cm Depth Average WFV (%)"].tolist()
        prep_data = df['Precipitation Precip (mm)'].tolist()

        for date,soil,prep in zip(dates,soil_datas,prep_data):
            if soil == 'No Data': self.no_data_dates.append(date)
            else:
                self.avaliable_data_dates.append(date)
                self.avaliable_datas.append(soil)
                self.avaliable_prep_data.append(prep)

    
                    

