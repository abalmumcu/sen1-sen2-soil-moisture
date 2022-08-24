from utils import dataset
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
from functools import reduce
warnings.filterwarnings("ignore")


csv_path = "/Users/alperbalmumcu/Github/sen1-sen2-soil-moisture/RISMA/dataset/datasets/"
prename = "Manitoba_Station_"

soil_list, prep_list,date_list = dataset.get_soil_prep_date(csv_path,prename)
df = dataset.merge_datasets_wdates(date_list,soil_list,prep_list)


plt.figure(figsize=(10,8))
plt.plot(df.Date,df['Soil_10'],color="blue",label="Soil / Station 1")
ax = plt.gca()
## plt.savefig("/content/drive/MyDrive/RISMA_full/Risma_Dataset/Manitoba_Station2_DailyData_[2016-02-29_2021-10-01].png")
plt.show()