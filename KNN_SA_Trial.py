# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import os


print(os.listdir())

data = pd.read_csv('./Data/Ariba_SV_Anomalies_Extract_V1_Q1_2018_20180412.csv')
print(data.info())
print(data.head())

#Deleting the Payment Date Fiscal year column.
data.drop('[INV]Payment Date (Fiscal Year)', axis=1, inplace=True)

print(list(data['[INV]Company Site (Division)'].unique()))
print(len(list(data['[INV]Company Site (Division)'].unique())))

print(len(set(data['[INV]Company Site (OU Description)'])) == 
      len(set(data['[INV]Company Site (Operating Unit Id)'])))

#Dropping [INV]Company Site (OU Description)
data.drop('[INV]Company Site (OU Description)', axis=1, inplace=True)

print(data.shape[1])

rules_columns=['sum(PO line quantity)', 'sum(Voucher Quantity)', 
              'sum(PO Amount)', 'sum(Voucher Amount)',
              '[PO]UOM (Unit of Measure)', 
              '[INV]Unit Of Measure (Unit of Measure)',
               'sum(Price Variance Cost)',
               '[INV]Voucher Create Date (Date)',
               '[PO]PO Create Date (Date)',
               '[INV]Voucher Invoice Date (Date)',
               '[INV]Voucher Received Date B (Date)',
               '[INV]Payment Date (Date)']


rules_data = data[rules_columns]

print(rules_data.info())

print(rules_data.head())

rules_data['[INV]Voucher Create Date (Date)'] = \
     rules_data['[INV]Voucher Create Date (Date)'].str.replace('-','')
     
rules_data['[INV]Voucher Create Date (Date)'] = \
     pd.to_numeric(rules_data['[INV]Voucher Create Date (Date)'].str.replace('\/',''))     

'[INV]Voucher Create Date (Date)'
rules_data['[PO]PO Create Date (Date)'] = \
     rules_data['[PO]PO Create Date (Date)'].str.replace('-','')
     
rules_data['[PO]PO Create Date (Date)'] = \
     pd.to_numeric(rules_data['[PO]PO Create Date (Date)'].str.replace('\/',''))     

rules_data['[INV]Voucher Invoice Date (Date)'] = \
     rules_data['[INV]Voucher Invoice Date (Date)'].str.replace('-','')
     
rules_data['[INV]Voucher Invoice Date (Date)'] = \
     pd.to_numeric(rules_data['[INV]Voucher Invoice Date (Date)'].str.replace('\/',''))     

rules_data['[INV]Voucher Received Date B (Date)'] = \
     rules_data['[INV]Voucher Received Date B (Date)'].str.replace('-','')
     
rules_data['[INV]Voucher Received Date B (Date)'] = \
     pd.to_numeric(rules_data['[INV]Voucher Received Date B (Date)'].str.replace('\/',''))  
     
rules_data['[INV]Payment Date (Date)'] = rules_data['[INV]Payment Date (Date)'].str.replace('Unclassified', '0')     

rules_data['[INV]Payment Date (Date)'] = \
     rules_data['[INV]Payment Date (Date)'].str.replace('-','')
     
rules_data['[INV]Payment Date (Date)'] = \
     pd.to_numeric(rules_data['[INV]Payment Date (Date)'].str.replace('/', ''))
     
print(rules_data.head())     

rules_data['vc-vr'] = rules_data['[INV]Voucher Create Date (Date)'] - \
                             rules_data['[INV]Voucher Received Date B (Date)']
                             
rules_data['vr-vi'] = rules_data['[INV]Voucher Received Date B (Date)'] - \
                              rules_data['[INV]Voucher Invoice Date (Date)']
                              
rules_data['vr-po'] = rules_data['[INV]Voucher Received Date B (Date)'] - \
                                       rules_data['[PO]PO Create Date (Date)']
                                       
rules_data['vi-po'] = rules_data['[INV]Voucher Invoice Date (Date)'] - \
                                        rules_data['[PO]PO Create Date (Date)']                                     

rules_data.drop(['[INV]Voucher Create Date (Date)',
               '[PO]PO Create Date (Date)',
               '[INV]Voucher Invoice Date (Date)',
               '[INV]Voucher Received Date B (Date)',
               '[INV]Payment Date (Date)'], axis=1, inplace=True)

print(rules_data.info())

print(rules_data['[PO]UOM (Unit of Measure)'].unique())

def find_difference_uom(uom_v, uom_p):
    return sum(1 for ch1, ch2 in zip(uom_v,uom_p) if ch1 != ch2) + abs(len(uom_v) - len(uom_p))

rules_data['v_uom-p_uom'] = rules_data[['[INV]Unit Of Measure (Unit of Measure)', 
                                     '[PO]UOM (Unit of Measure)' ]].apply(lambda x: find_difference_uom(x['[INV]Unit Of Measure (Unit of Measure)'], x['[PO]UOM (Unit of Measure)']), axis=1)

rules_data.drop(['[PO]UOM (Unit of Measure)','[INV]Unit Of Measure (Unit of Measure)'], axis=1, inplace=True)


print(rules_data.info())

rules_data['sum(PO line quantity)'] = \
     rules_data['sum(PO line quantity)'].str.replace(',','')
rules_data['sum(Voucher Quantity)'] = \
     rules_data['sum(Voucher Quantity)'].str.replace(',','')
rules_data['sum(PO Amount)'] = \
     rules_data['sum(PO Amount)'].str.replace(',','')
rules_data['sum(Voucher Amount)'] = \
     rules_data['sum(Voucher Amount)'].str.replace(',','')
rules_data['sum(Price Variance Cost)'] = \
     rules_data['sum(Price Variance Cost)'].str.replace(',','')     


rules_data['sum(PO line quantity)'] = rules_data['sum(PO line quantity)'].apply(float)
rules_data['sum(Voucher Quantity)'] = rules_data['sum(Voucher Quantity)'].apply(float)
rules_data['sum(PO Amount)'] = rules_data['sum(PO Amount)'].apply(float)
rules_data['sum(Voucher Amount)'] = rules_data['sum(Voucher Amount)'].apply(float)
rules_data['sum(Price Variance Cost)'] = rules_data['sum(Price Variance Cost)'].apply(float)

scaler = StandardScaler()
rules_data = scaler.fit_transform(rules_data)


lof = LocalOutlierFactor(n_neighbors=100)

lof.fit(rules_data)
y_pred = lof.fit_predict(rules_data)

num_oultiers = sum(1 for res in y_pred if res < 0)
print(num_outliers)