import pandas as pd
import yaml, os
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

######## Notes: Correct path. Output intermediate files. Github.
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Access the raw data directory
data_dir = config["raw_data"]["nhts"]
processed_dir = config["processed_data"]["nhts"]

adjweights = pd.read_csv(os.path.join(data_dir,"2017","county_wt_adj.csv"))

########### Read household data: 2017
hh = pd.read_csv(os.path.join(data_dir,"2017","hhpub.csv")) # 129696 households
hh['HOUSEID'] = hh['HOUSEID'].apply(str)
hh_nys = pd.read_csv(os.path.join(data_dir,"2017","nys_household.csv"))
hh_nys['HOUSEID'] = hh_nys['HOUSEID'].apply(str)
hh_nys = hh_nys[hh_nys.HHSTATE == 'NY']
hh_nys = hh_nys.drop(columns='WTHHFIN')
hh_nys = hh_nys.rename(columns={'WTHHFIN_ADJ':'WTHHFIN'})
NYChouseholds = list(set(hh_nys.loc[hh_nys['CNTYFIPS'].isin([36047,36061,36081,36085,36005]),'HOUSEID']))
print(len(hh_nys))

# remove public NYS households
NYShouseholds = list(set(hh_nys.HOUSEID))
hh = hh[~hh.HOUSEID.isin(NYShouseholds)]

# remvoe zero weight households
zeroWeight = hh[hh.WTHHFIN==0].HOUSEID
hh = hh[~hh.HOUSEID.isin(zeroWeight)]
print('Non-NYS households: '+str(len(hh)))
zeroWeight_nys = hh_nys[hh_nys.WTHHFIN==0].HOUSEID
hh_nys = hh_nys[~hh_nys.HOUSEID.isin(zeroWeight_nys)]
print('NYS households: '+str(len(hh_nys)))

# urban class
hh['URBRUR'] = hh['URBRUR'].map({1:'Urban',2:'Rural'})
hh_nys['URBRUR'] = hh_nys['URBRUR'].map({1:'Urban',2:'Rural'})
hh_nys['URBRUR_NYC'] = 'non-NYC Rural'
hh_nys.loc[(~hh_nys.HOUSEID.isin(NYChouseholds))&(hh_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
hh_nys.loc[(hh_nys.HOUSEID.isin(NYChouseholds)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural households: '+str(len(hh_nys[hh_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban households: '+str(len(hh_nys[hh_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC households: '+str(len(hh_nys[hh_nys.URBRUR_NYC=='NYC'])))

hh.to_csv(os.path.join(processed_dir,"hh.csv"),index=False)
hh_nys.to_csv(os.path.join(processed_dir,"hh_nys.csv"),index=False)

########### Read household data: 2009
hh2009 = pd.read_csv(os.path.join(data_dir,"2009National","HHV2PUB.CSV"))
hh2009['HOUSEID'] = hh2009['HOUSEID'].apply(str)
print('Public households: '+str(len(hh2009))) # 150147
hh2009_nys = pd.read_sas(os.path.join(data_dir,"2009NYSDOT","ny_hhv2.sas7bdat"), format = 'sas7bdat', encoding="utf-8")
hh2009_nys['HOUSEID'] = hh2009_nys['HOUSEID'].apply(str)
hh2009_nys = hh2009_nys[hh2009_nys['HHSTATE']=='NY'] # filter out other states!
# missing variable in nys
hh2009_nys = pd.merge(hh2009_nys,hh2009[['HOUSEID','HBPPOPDN']],on='HOUSEID',how='left')

urbanClass = pd.read_csv(os.path.join(data_dir,"2009National","cen10pub","cen10pub.csv"))
urbanClass['HOUSEID'] = urbanClass['HOUSEID'].apply(str)# re

# remove public NYS households
NYShouseholds2009 = list(set(hh2009_nys.HOUSEID))
domainhh = pd.read_sas(os.path.join(data_dir,"2009NYSDOT","strat_domainhh.sas7bdat"), format = 'sas7bdat', encoding="utf-8")
NYChouseholds2009 = list(set(domainhh.loc[domainhh['DOMAIN01'].isin([812,813,814,815,816]),'HOUSEID']))
hh2009 = hh2009[~hh2009.HOUSEID.isin(NYShouseholds2009)]

# remvoe zero weight households
zeroWeight2009 = hh2009[hh2009.WTHHFIN==0].HOUSEID
hh2009 = hh2009[~hh2009.HOUSEID.isin(zeroWeight2009)]
print('Non-NYS households: '+str(len(hh2009)))
zeroWeight2009_nys = hh2009_nys[hh2009_nys.WTHHFIN==0].HOUSEID
hh2009_nys = hh2009_nys[~hh2009_nys.HOUSEID.isin(zeroWeight2009_nys)]
print('NYS households: '+str(len(hh2009_nys)))

# to integer
cols_filtered = hh2009_nys.columns.difference(['HH_CBSA','HHC_MSA','HH_PERSWRKR','HH_RACOS',
                                               'HOMEOWOS','HHSTATE','CNTYNYOS','HOMETYOS','LANDCELL',
                                               'HOUSEID'])
hh2009_nys[cols_filtered] = hh2009_nys[cols_filtered].astype(int)

# urban class
hh2009 = hh2009.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
hh2009 = pd.merge(hh2009,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
hh2009 = hh2009.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
hh2009['URBRUR'] = hh2009['URBRUR'].map({1:'Urban',2:'Rural'})
hh2009_nys = hh2009_nys.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
hh2009_nys = pd.merge(hh2009_nys,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
hh2009_nys = hh2009_nys.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
hh2009_nys['URBRUR'] = hh2009_nys['URBRUR'].map({1:'Urban',2:'Rural'})
hh2009_nys['URBRUR_NYC'] = 'non-NYC Rural'
hh2009_nys.loc[(~hh2009_nys.HOUSEID.isin(NYChouseholds2009))&(hh2009_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
hh2009_nys.loc[(hh2009_nys.HOUSEID.isin(NYChouseholds2009)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural households: '+str(len(hh2009_nys[hh2009_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban households: '+str(len(hh2009_nys[hh2009_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC households: '+str(len(hh2009_nys[hh2009_nys.URBRUR_NYC=='NYC'])))

hh2009.to_csv(os.path.join(processed_dir,"hh2009.csv"),index=False)
hh2009_nys.to_csv(os.path.join(processed_dir,"hh2009_nys.csv"),index=False)

########## Read person data: 2017
person = pd.read_csv(os.path.join(data_dir,"2017","perpub.csv"))
person['HOUSEID'] = person['HOUSEID'].apply(str)
print('Public persons ge 5: '+str(len(person)))
person_nys = pd.read_csv(os.path.join(data_dir,"2017","nys_person.csv"))
person_nys['HOUSEID'] = person_nys['HOUSEID'].apply(str)
person_nys = person_nys[person_nys.HHSTATE == 'NY']

# remove younger than 5
person = person[person.R_AGE_IMP>=5]
person_nys = person_nys[person_nys.R_AGE_IMP>=5]

# remove public NYS households
person = person[~person.HOUSEID.isin(NYShouseholds)]

# remvoe zero weight households and persons
person = person[~person.HOUSEID.isin(zeroWeight)]
person = person[person.WTPERFIN!=0]
print('Non-NYS persons: '+str(len(person)))
person_nys = person_nys[~person_nys.HOUSEID.isin(zeroWeight_nys)]
person_nys = person_nys[person_nys.WTPERFIN!=0]
print('NYS persons: '+str(len(person_nys)))

# urban class
person['URBRUR'] = person['URBRUR'].map({1:'Urban',2:'Rural'})
person_nys['URBRUR'] = person_nys['URBRUR'].map({1:'Urban',2:'Rural'})
person_nys['URBRUR_NYC'] = 'non-NYC Rural'
person_nys.loc[(~person_nys.HOUSEID.isin(NYChouseholds))&(person_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
person_nys.loc[(person_nys.HOUSEID.isin(NYChouseholds)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural persons: '+str(len(person_nys[person_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban persons: '+str(len(person_nys[person_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC persons: '+str(len(person_nys[person_nys.URBRUR_NYC=='NYC'])))

person.to_csv(os.path.join(processed_dir,"person.csv"),index=False)
person_nys.to_csv(os.path.join(processed_dir,"person_nys.csv"),index=False)

############### Read person file: 2009
person2009 = pd.read_csv(os.path.join(data_dir,"2009National","PERV2PUB.CSV"))
person2009['HOUSEID'] = person2009['HOUSEID'].apply(str)
person2009['PERSONID'] = person2009['PERSONID'].apply(int)
print('public persons ge 5: '+str(len(person2009)))
person2009_nys = pd.read_sas(os.path.join(data_dir,"2009NYSDOT","ny_persv2.sas7bdat"), format = 'sas7bdat', encoding='ISO-8859-1')
person2009_nys['HOUSEID'] = person2009_nys['HOUSEID'].apply(str)
person2009_nys['PERSONID'] = person2009_nys['PERSONID'].apply(int)

person2009_nys = person2009_nys[person2009_nys['HHSTATE']=='NY'] # filter out other states!
person2009_nys = person2009_nys.drop(columns=['WTPERFINN'])
# missing variable in nys
person2009_nys = pd.merge(person2009_nys,person2009[['HOUSEID','PERSONID','HBPPOPDN']],on=['HOUSEID','PERSONID'],how='left')

# remove younger than 5
person2009 = person2009[person2009.R_AGE>=5]
person2009_nys = person2009_nys[person2009_nys.R_AGE>=5]

# remove public NYS households
person2009 = person2009[~person2009.HOUSEID.isin(NYShouseholds2009)]

# remvoe zero weight households and persons
person2009 = person2009[~person2009.HOUSEID.isin(zeroWeight2009)]
person2009 = person2009[person2009.WTPERFIN!=0]
print('Non-NYS persons: '+str(len(person2009)))
person2009_nys = person2009_nys[~person2009_nys.HOUSEID.isin(zeroWeight2009_nys)]
person2009_nys = person2009_nys[person2009_nys.WTPERFIN!=0]
print('NYS persons: '+str(len(person2009_nys)))

# to integer
not_integer = person2009_nys.columns[person2009_nys.apply(pd.to_numeric, errors='coerce').isnull().any()].tolist()
cols_filtered = list(person2009_nys.columns.difference(not_integer))
person2009_nys[cols_filtered] = person2009_nys[cols_filtered].astype(np.int64)
person2009_nys['HOUSEID'] = person2009_nys['HOUSEID'].apply(str)

# urban class
person2009 = person2009.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
person2009 = pd.merge(person2009,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
person2009 = person2009.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
person2009['URBRUR'] = person2009['URBRUR'].map({1:'Urban',2:'Rural'})
person2009_nys = person2009_nys.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
person2009_nys = pd.merge(person2009_nys,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
person2009_nys = person2009_nys.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
person2009_nys['URBRUR'] = person2009_nys['URBRUR'].map({1:'Urban',2:'Rural'})

person2009_nys['URBRUR_NYC'] = 'non-NYC Rural'
person2009_nys.loc[(~person2009_nys.HOUSEID.isin(NYChouseholds2009))&(person2009_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
person2009_nys.loc[(person2009_nys.HOUSEID.isin(NYChouseholds2009)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural persons: '+str(len(person2009_nys[person2009_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban persons: '+str(len(person2009_nys[person2009_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC persons: '+str(len(person2009_nys[person2009_nys.URBRUR_NYC=='NYC'])))

person2009.to_csv(os.path.join(processed_dir,"person2009.csv"),index=False)
person2009_nys.to_csv(os.path.join(processed_dir,"person2009_nys.csv"),index=False)

###### Read vehicle file: 2017
veh = pd.read_csv(os.path.join(data_dir,"2017","vehpub.csv"))
veh['HOUSEID'] = veh['HOUSEID'].apply(str)
print('Public vehicles: '+str(len(veh)))
# veh = veh[veh['PERSONID'].isin([1,2,3,4,5,6,7,8,9,10])]
veh_nys = pd.read_csv(os.path.join(data_dir,"2017","nys_vehicle.csv"))
veh_nys['HOUSEID'] = veh_nys['HOUSEID'].apply(str)
veh_nys = veh_nys[veh_nys.HHSTATE == 'NY']
veh_nys = veh_nys.drop(columns='WTHHFIN')
veh_nys = veh_nys.rename(columns={'WTHHFIN_ADJ':'WTHHFIN'})

# remove public NYS households
veh = veh[~veh.HOUSEID.isin(NYShouseholds)]

# remvoe zero weight households
veh = veh[~veh.HOUSEID.isin(zeroWeight)]
print('Non-NYS vehicles: '+str(len(veh)))
veh_nys = veh_nys[~veh_nys.HOUSEID.isin(zeroWeight_nys)]
print('NYS vehicles: '+str(len(veh_nys)))

# urban class
veh['URBRUR'] = veh['URBRUR'].map({1:'Urban',2:'Rural'})
veh_nys['URBRUR'] = veh_nys['URBRUR'].map({1:'Urban',2:'Rural'})
veh_nys['URBRUR_NYC'] = 'non-NYC Rural'
veh_nys.loc[(~veh_nys.HOUSEID.isin(NYChouseholds))&(veh_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
veh_nys.loc[(veh_nys.HOUSEID.isin(NYChouseholds)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural vehs: '+str(len(veh_nys[veh_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban vehs: '+str(len(veh_nys[veh_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC vehs: '+str(len(veh_nys[veh_nys.URBRUR_NYC=='NYC'])))

veh.to_csv(os.path.join(processed_dir,"veh.csv"),index=False)
veh_nys.to_csv(os.path.join(processed_dir,"veh_nys.csv"),index=False)

######## Read veh file 2009
veh2009 = pd.read_csv(os.path.join(data_dir,"2009National","VEHV2PUB.CSV"))
veh2009['HOUSEID'] = veh2009['HOUSEID'].apply(str)
veh2009['VEHID'] = veh2009['VEHID'].astype(int).apply(str)
print('Public vehicles: '+str(len(veh2009)))
# veh2009 = veh2009[veh2009['PERSONID'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14])]
veh2009_nys = pd.read_sas(os.path.join(data_dir,"2009NYSDOT","ny_vehv2.sas7bdat"), format = 'sas7bdat', encoding='ISO-8859-1')
veh2009_nys['HOUSEID'] = veh2009_nys['HOUSEID'].apply(str)
veh2009_nys['VEHID'] = veh2009_nys['VEHID'].astype(int).apply(str)
veh2009_nys = veh2009_nys[veh2009_nys['HHSTATE']=='NY'] # filter out other states!

# add variable BESTMILE from public file
veh2009_nys = pd.merge(veh2009_nys,veh2009[['HOUSEID','VEHID','BESTMILE']],on=['HOUSEID','VEHID'],how='left').fillna(0)

# remove public NYS households
veh2009 = veh2009[~veh2009.HOUSEID.isin(NYShouseholds2009)]

# remvoe zero weight households
veh2009 = veh2009[~veh2009.HOUSEID.isin(zeroWeight2009)]
print('Non-NYS vehicles: '+str(len(veh2009)))
veh2009_nys = veh2009_nys[~veh2009_nys.HOUSEID.isin(zeroWeight2009_nys)]
print('NYS vehicles: '+str(len(veh2009_nys)))

# to integer
cols_filtered = veh2009_nys.columns.difference(['HH_CBSA','HHC_MSA','LANDCELL','HHSTATE','MAKENAME','MODLNAME','VEHTYOS', 
                                               'HOUSEID'])
veh2009_nys[cols_filtered] = veh2009_nys[cols_filtered].astype(int)

# urban class
veh2009 = veh2009.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
veh2009 = pd.merge(veh2009,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
veh2009 = veh2009.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
veh2009['URBRUR'] = veh2009['URBRUR'].map({1:'Urban',2:'Rural'})
veh2009_nys = veh2009_nys.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
veh2009_nys = pd.merge(veh2009_nys,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
veh2009_nys = veh2009_nys.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
veh2009_nys['URBRUR'] = veh2009_nys['URBRUR'].map({1:'Urban',2:'Rural'})
veh2009_nys['URBRUR_NYC'] = 'non-NYC Rural'
veh2009_nys.loc[(~veh2009_nys.HOUSEID.isin(NYChouseholds2009))&(veh2009_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
veh2009_nys.loc[(veh2009_nys.HOUSEID.isin(NYChouseholds2009)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural households: '+str(len(veh2009_nys[veh2009_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban households: '+str(len(veh2009_nys[veh2009_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC households: '+str(len(veh2009_nys[veh2009_nys.URBRUR_NYC=='NYC'])))

veh2009.to_csv(os.path.join(processed_dir,"veh2009.csv"),index=False)
veh2009_nys.to_csv(os.path.join(processed_dir,"veh2009_nys.csv"),index=False)

########### Read trip data 2017
trip = pd.read_csv(os.path.join(data_dir,"2017","trippub.csv"))
print('Public trip: '+str(len(trip)))
trip['HOUSEID'] = trip['HOUSEID'].apply(str)
trip_nys = pd.read_csv(os.path.join(data_dir,"2017","nys_trip.csv"))
trip_nys['HOUSEID'] = trip_nys['HOUSEID'].apply(str)

# remove age
trip = trip[trip.R_AGE_IMP>=5]
trip_nys = trip_nys[trip_nys.R_AGE_IMP>=5]

# remove public NYS households
trip = trip[~trip.HOUSEID.isin(NYShouseholds)]

# remvoe zero weight households and persons
trip = trip[~trip.HOUSEID.isin(zeroWeight)]
trip = trip[trip.WTTRDFIN!=0]
print('Non-NYS trip: '+str(len(trip)))
trip_nys = trip_nys[~trip_nys.HOUSEID.isin(zeroWeight_nys)]
trip_nys = trip_nys[trip_nys.WTTRDFIN!=0]
print('NYS trip: '+str(len(trip_nys)))

# fix some trips
trip_nys.loc[(trip_nys.HOUSEID=='40499852')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==1),'TRPMILAD']=1
trip_nys.loc[(trip_nys.HOUSEID=='40499852')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==2),'TRVLCMIN']=30
trip_nys.loc[(trip_nys.HOUSEID=='40289409')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==2),'TRPMILAD']=2
trip_nys.loc[(trip_nys.HOUSEID=='40289409')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==3),'TRPMILAD']=2
trip_nys.loc[(trip_nys.HOUSEID=='40289409')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==4),'TRPMILAD']=5
trip_nys.loc[(trip_nys.HOUSEID=='40289409')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==2),'TRVLCMIN']=10
trip_nys.loc[(trip_nys.HOUSEID=='40289409')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==3),'TRVLCMIN']=15
trip_nys.loc[(trip_nys.HOUSEID=='40289409')&(trip_nys.PERSONID==2)&(trip_nys.TDTRPNUM==4),'TRVLCMIN']=15

# urban class
trip['URBRUR'] = trip['URBRUR'].map({1:'Urban',2:'Rural'})
trip_nys['URBRUR'] = trip_nys['URBRUR'].map({1:'Urban',2:'Rural'})
trip_nys['URBRUR_NYC'] = 'non-NYC Rural'
trip_nys.loc[(~trip_nys.HOUSEID.isin(NYChouseholds))&(trip_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
trip_nys.loc[(trip_nys.HOUSEID.isin(NYChouseholds)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural trips: '+str(len(trip_nys[trip_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban trips: '+str(len(trip_nys[trip_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC trips: '+str(len(trip_nys[trip_nys.URBRUR_NYC=='NYC'])))

trip.to_csv(os.path.join(processed_dir,"trip.csv"),index=False)
trip_nys.to_csv(os.path.join(processed_dir,"trip_nys.csv"),index=False)

############ Read trip data 2009
trip2009 = pd.read_csv(os.path.join(data_dir,"2009National","DAYV2PUB.CSV"))
trip2009['HOUSEID'] = trip2009['HOUSEID'].apply(str)
print('Public trip: '+str(len(trip2009)))
trip2009_nys = pd.read_sas(os.path.join(data_dir,"2009NYSDOT","ny_dtrpv2.sas7bdat"), format = 'sas7bdat', encoding='ISO-8859-1')
trip2009_nys['HOUSEID'] = trip2009_nys['HOUSEID'].apply(str)
trip2009_nys = trip2009_nys[trip2009_nys['HHSTATE']=='NY'] # filter out other states!

# remove age
trip2009 = trip2009[trip2009.R_AGE>=5]
trip2009_nys = trip2009_nys[trip2009_nys.R_AGE>=5]

# remove public NYS households
trip2009 = trip2009[~trip2009.HOUSEID.isin(NYShouseholds2009)]

# remvoe zero weight households and persons
trip2009 = trip2009[~trip2009.HOUSEID.isin(zeroWeight2009)]
trip2009 = trip2009[trip2009.WTTRDFIN!=0]
print('Non-NYS trip: '+str(len(trip2009)))
trip2009_nys = trip2009_nys[~trip2009_nys.HOUSEID.isin(zeroWeight2009_nys)]
trip2009_nys = trip2009_nys[trip2009_nys.WTTRDFIN!=0]
print('NYS trip: '+str(len(trip2009_nys)))

# to integer
not_integer = trip2009_nys.columns[trip2009_nys.apply(pd.to_numeric, errors='coerce').isnull().any()].tolist()
cols_filtered = list(trip2009_nys.columns.difference(not_integer))
trip2009_nys[cols_filtered] = trip2009_nys[cols_filtered].astype(np.int64)
trip2009_nys['HOUSEID'] = trip2009_nys['HOUSEID'].apply(str)

# urban class
trip2009 = trip2009.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
trip2009 = pd.merge(trip2009,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
trip2009 = trip2009.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
trip2009['URBRUR'] = trip2009['URBRUR'].map({1:'Urban',2:'Rural'})
trip2009_nys = trip2009_nys.drop(['URBAN','URBANSIZE','URBRUR'], axis=1)
trip2009_nys = pd.merge(trip2009_nys,urbanClass[['HOUSEID','URBAN10','URBSIZE10','URBRUR10']],how='left',on='HOUSEID')
trip2009_nys = trip2009_nys.rename(columns={'URBAN10':'URBAN','URBSIZE10':'URBANSIZE','URBRUR10':'URBRUR'})
trip2009_nys['URBRUR'] = trip2009_nys['URBRUR'].map({1:'Urban',2:'Rural'})
trip2009_nys['URBRUR_NYC'] = 'non-NYC Rural'
trip2009_nys.loc[(~trip2009_nys.HOUSEID.isin(NYChouseholds2009))&(trip2009_nys.URBRUR=='Urban'),'URBRUR_NYC'] = 'non-NYC Urban'
trip2009_nys.loc[(trip2009_nys.HOUSEID.isin(NYChouseholds2009)),'URBRUR_NYC'] = 'NYC'
print('non-NYC Rural households: '+str(len(trip2009_nys[trip2009_nys.URBRUR_NYC=='non-NYC Rural'])))
print('non-NYC Urban households: '+str(len(trip2009_nys[trip2009_nys.URBRUR_NYC=='non-NYC Urban'])))
print('NYC households: '+str(len(trip2009_nys[trip2009_nys.URBRUR_NYC=='NYC'])))

trip2009.to_csv(os.path.join(processed_dir,"trip2009.csv"),index=False)
trip2009_nys.to_csv(os.path.join(processed_dir,"trip2009_nys.csv"),index=False)