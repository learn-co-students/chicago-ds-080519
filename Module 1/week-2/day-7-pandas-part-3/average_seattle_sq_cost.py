#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd

sales_df = pd.read_csv('data/EXTR_RPSale.csv')
sales_df = sales_df[['Major', 'Minor', 'DocumentDate', 'SalePrice']]

bldg_df = pd.read_csv('data/EXTR_ResBldg.csv')
bldg_df = bldg_df[['Major', 'Minor', 'SqFtTotLiving', 'ZipCode']]

sales_df['Minor'] = pd.to_numeric(sales_df['Minor'],errors='coerce')
sales_df['Major'] = pd.to_numeric(sales_df['Major'],errors='coerce')

sales_data = pd.merge(sales_df, bldg_df, on=['Major', 'Minor'])

sales_data = sales_data.loc[~sales_data['ZipCode'].isna(), :]


cleaned_data_SP = sales_data.loc[(sales_data['SalePrice']>0)]
cleaned_data_sql = cleaned_data_SP.loc[(cleaned_data_SP['SqFtTotLiving']>0)]

cleaned_data_sql['ZipCode2'] = cleaned_data_sql['ZipCode'].map(lambda x: str(x)[:5])

cleaned_data_sql['PricePerSqFt'] = cleaned_data_sql.SalePrice/cleaned_data_sql.SqFtTotLiving

# get only 2019 sales
data2019 = cleaned_data_sql.loc[cleaned_data_sql.DocumentDate.str[-4:] == '2019']

Zipcodes = [98101, 98102, 98103, 98104, 98105, 98106, 98107, 98108, 98109, 98112, 98115, 98116, 98117, 98118, 98119, 98121, 98122, 98125, 98126, 98133, 98134, 98136, 98144, 98146, 98154, 98164, 98174, 98177, 98178, 98195, 98199]

zipseries = pd.Series(Zipcodes)
test = zipseries.astype(str)
ready_data = data2019.loc[data2019.ZipCode2.isin(test)]

average_price_psf=ready_data.PricePerSqFt.mean()
print(average_price_psf)

