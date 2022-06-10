import pickle
import inflection
import pandas as pd
import numpy as np
import math 
import datetime 


# store_type, competition_distance, competition_time_month, promo_time_week


class Rossman(object):
    
    def __init__(self):

        self.home_path=''

        self.encoding_competition_distance_scaler   = pickle.load(open(self.home_path + 'parameter/encoding_competition_distance_scaler.pkl','rb'))
        self.encoding_competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/encoding_competition_time_month_scaler.pkl','rb'))
        self.encoding_promo_time_week_scaler        = pickle.load(open(self.home_path + 'parameter/encoding_promo_time_week_scaler.pkl','rb'))
        self.encoding_year_scaler                   = pickle.load(open(self.home_path + 'parameter/encoding_year_scaler.pkl','rb'))
        self.store_type_scaler                      = pickle.load(open(self.home_path + 'parameter/store_type_scaler.pkl','rb'))
        
    def data_cleaning(self,df1):
        
        ## 1.1 Rename Columns
        cols_new = ['store', 'day_of_week', 'date', 'open', 'promo',
               'state_holiday', 'school_holiday', 'store_type', 'assortment',
               'competition_distance', 'competition_open_since_month',
               'competition_open_since_year', 'promo2', 'promo2_since_week',
               'promo2_since_year', 'promo_interval']
        
        df1.columns = cols_new

        ## 1.3 Data Types
        df1['date'] = pd.to_datetime(df1['date'])

        ## 1.5 Fill NA
        #competition_distance - Se for tao longe que nao foi computado, colocar um valor mto grande!
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000 if math.isnan(x) else x)

        #competition_open_since_month  
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan( x['competition_open_since_month']) else x['competition_open_since_month'],axis=1)

        #competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan( x['competition_open_since_year']) else x['competition_open_since_year'],axis=1)

        #promo2_since_week    
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan( x['promo2_since_week']) else x['promo2_since_week'],axis=1)

        #promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan( x['promo2_since_year']) else x['promo2_since_year'],axis=1)

        #promo_interval
        month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5: 'May',6: 'Jun',7: 'Jul',8: 'Aug',9: 'Sept',10:'Oct',11:'Nov',12:'Dec'}

        df1['promo_interval'].fillna(0,inplace=True)
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo'] = df1[['promo_interval','month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0,axis=1) 

        ## 1.6 Change Types
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)


        #df1['store_type'] = df1['store_type'].astype(int)
        #df1['competition_distance'] = df1['competition_distance'].astype(int)
        #df1['competition_time_month'] =
        #df1['promo_time_week'] = 

        # store_type, competition_distance, competition_time_month, promo_time_week

        
        return df1

    def feature_engineering(self,df1):
        
        ## year
        df1['year'] = df1['date'].dt.year

        ## month
        df1['month'] = df1['date'].dt.month

        ## day
        df1['day'] = df1['date'].dt.day

        ## week of year
        df1['week_of_year'] = df1['date'].dt.isocalendar().week

        # year week
        df1['year_week'] = df1['date'].dt.strftime('%Y-%W')

        #competition_since
        df1['competition_since'] = df1.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],month=x['competition_open_since_month'],day=1),axis=1)

        df1['competition_time_month'] = ((df1['date'] - df1['competition_since'] ) /30).apply(lambda x: x.days).astype(int)

        #promo_since
        df1['promo_since'] = df1['promo2_since_year'].astype(str) + '-' + df1['promo2_since_week'].astype(str)
        df1['promo_since'] = df1['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

        df1['promo_time_week'] = ((df1['date'] - df1['promo_since'] ) / 7).apply(lambda x: x.days).astype(int)

        #assortment
        df1['assortment'] = df1['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        #state_holiday
        df1['state_holiday'] = df1['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        ## 3.1 Filtragem das Linhas
        df1 = df1[df1['open'] != 0]

        ## 3.2 Seleção das Colunas
        cols_drop = ['open','promo_interval','month_map']
        df1 = df1.drop(cols_drop,axis=1)
        
        return df1
    
    
    def data_preparation(self,df5):
        ## 5.2 Reescaling
        # competition_distance - ROBUST SCALLER
        df5['competition_distance'] = self.encoding_competition_distance_scaler.transform(df5[['competition_distance']].values)
    
        # competition_time_month - ROBUST SCALLER
        df5['competition_time_month'] = self.encoding_competition_time_month_scaler.transform(df5[['competition_time_month']].values)

        # promo_time_week -  MIN MAX - MinMaxScaler
        df5['promo_time_week'] = self.encoding_promo_time_week_scaler.transform(df5[['promo_time_week']].values)

        # year - MIN MAX
        df5['year'] = self.encoding_year_scaler.transform(df5[['year']].values)

        ## 5.3 Transformação
        ### 5.3.1 Encoding
        #state_holiday - regular / public / easter / christmas - One Hot Encoding
        df5 = pd.get_dummies(df5,prefix=['state_holiday'],columns=['state_holiday'])

        #store_type - a b c -> Label Encoding
        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])

        #assortment -> basic - extended - extra -> Ordinal Encoding
        assortment_dict = {'basic':1,'extra':2,'extended':3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        ### 5.3.3 Nature Transformation
        ## month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2* np.pi/12)) )
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2* np.pi/12)) )

        ## day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2* np.pi/30)) )
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2* np.pi/30)) )


        ## week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2* np.pi/52)) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2* np.pi/52)) )

        ## day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2* np.pi/7)) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2* np.pi/7)) )
        
        cols_final = ['store',
         'promo',
         'store_type',
         'assortment',
         'competition_distance',
         'competition_open_since_month',
         'competition_open_since_year',
         'promo2',
         'promo2_since_week',
         'promo2_since_year',
         'competition_time_month',
         'promo_time_week',
         'month_sin',              
         'month_cos',
         'day_sin',
         'day_cos',
         'week_of_year_sin',
         'week_of_year_cos',
         'day_of_week_sin',
         'day_of_week_cos']
        
        return df5[cols_final]


    def get_prediction(self,model,original_data,test_data):
        # prediction
        pred = model.predict(test_data)

        # Join Predic into the original data

        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records',date_format='iso')