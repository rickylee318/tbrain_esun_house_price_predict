import pandas as pd
import numpy as np
from Config import config
import sys

class data_process:
    def __init__(self, config, datapath=''):
        self.datapath = datapath
        self.config = config
    def load_data(self):
        datapath = self.datapath
        test_datapath = 'test.csv'
        train_datapath = 'train.csv'
        print(test_datapath)
        #test data
        test = pd.read_csv(test_datapath)
        test.index = test.building_id
        test.drop(columns = ['building_id'], inplace=True)
        print(test.head(5))
        #train data
        train = pd.read_csv(train_datapath)
        train.index = train.building_id
        train.drop(columns = ['building_id'], inplace=True)
        print(train.head(5))
        #concate data
        df = pd.concat([train,test], axis=0, sort=False)
        self.data = df
        return True
        
    def numpy_triu1(self):          
        a = df.values
        r,c = np.triu_indices(a.shape[1],1)
        cols = df.columns
        nm = [cols[i]+"_"+cols[j] for i,j in zip(r,c)]
        return pd.DataFrame(a[:,r] - a[:,c], columns=nm, index = df.index)

    def numpy_triu2(self):          
        a = df.values
        r,c = np.triu_indices(a.shape[1],1)
        cols = df.columns
        nm = [cols[i]+"/"+cols[j] for i,j in zip(r,c)]
        return pd.DataFrame(a[:,r] / a[:,c], columns=nm, index = df.index)
    
    def feature_transform(self):
        data = self.data
        data = data.fillna(0)
        for transform in self.config.get_transforms():
            print(transform)
            if transform == 'txn_floor':
                data.drop(columns = ['txn_floor'], inplace=True)
                
            if transform == 'txn_dt_year':
                data['txn_dt_year'] = data.txn_dt // 365
                
            if transform == 'building_complete_dt_year':
                data['building_complete_dt_year'] = data.building_complete_dt // 365
                
            if transform == 'city_population':
                city_population = data[['city','town_population']].groupby('city').sum()
                li = []
                for i in data.index.tolist():
                    tmp = data.loc[i, 'city']
                    li.append(city_population.loc[tmp].iloc[0])
                data = pd.concat([data, pd.DataFrame(li, index=data.index, columns = ['city_population'])],axis=1)
                
            if transform == 'city_area':
                city_area = data[['city','town_area']].groupby('city').sum()
                li = []
                for i in data.index.tolist():
                    tmp = data.loc[i, 'city']
                    li.append(city_area.loc[tmp].iloc[0])
                data = pd.concat([data, pd.DataFrame(li, index=data.index, columns = ['city_area'])], axis=1)
                
            if transform == "city_population_density":
                data['city_population_density'] = data['city_population'] / data['city_area']
                
            if transform == 'log_total_price':
                data['log_total_price'] = np.log(data.total_price)
                
            if transform == 'doc_num':
                data['doc_num'] = data.doc_rate * data.city_population
                
            if transform == 'bachelor_num':
                data['bachelor_num'] = data.bachelor_rate * data.city_population
                
            if transform == 'highschool_num':
                data['highschool_num'] = data.highschool_rate * data.city_population
                
            if transform == 'jobschool_num':
                data['jobschool_num'] = data.jobschool_rate * data.city_population
                
            if transform == 'junior_num':
                data['junior_num'] = data.junior_rate * data.city_population
                
            if transform == 'elementary_num':
                data['elementary_num'] = data.elementary_rate * data.city_population
                
            if transform == 'born_num':
                data['born_num'] = data.born_rate * data.city_population
                
            if transform == 'death_num':
                data['death_num'] = data.death_rate * data.city_population
                
            if transform == 'marriage_num':
                data['marriage_num'] =data.marriage_rate * data.city_population
                
            if transform == 'divorce_num':
                data['divorce_num'] = data.divorce_rate * data.city_population
                
            if transform == 'town_income_median_mean':
                tmp = data[['town','village_income_median']].groupby('town').mean()
                li = []
                for i in data.index.tolist():
                    t = data.loc[i, 'town']
                    li.append(tmp.loc[t].iloc[0])
                data = pd.concat([data, pd.DataFrame(li, index=data.index, columns = ['town_income_median_mean'])], axis=1)
                
            if transform == 'town_income_median_min':
                tmp = data[['town','village_income_median']].groupby('town').min()
                li = []
                for i in data.index.tolist():
                    t = data.loc[i, 'town']
                    li.append(tmp.loc[t].iloc[0])
                data = pd.concat([data, pd.DataFrame(li, index=data.index, columns = ['town_income_median_min'])], axis=1)
                
            if transform == 'town_income_median_max':
                tmp = data[['town','village_income_median']].groupby('town').max()
                li = []
                for i in data.index.tolist():
                    t = data.loc[i, 'town']
                    li.append(tmp.loc[t].iloc[0])
                data = pd.concat([data, pd.DataFrame(li, index=data.index, columns = ['town_income_median_max'])], axis=1)
                
            if transform == 'town_income_median_median':
                tmp = data[['town','village_income_median']].groupby('town').median()
                li = []
                for i in data.index.tolist():
                    t = data.loc[i, 'town']
                    li.append(tmp.loc[t].iloc[0])
                data = pd.concat([data, pd.DataFrame(li, index=data.index, columns = ['town_income_median_median'])], axis=1)
                
        self.data = data
        
    def category(self):
        try:
            data = self.data
            data.city = data.city.astype('category')
            data.town = data.town.astype('category')
            data.village = data.village.astype('category')
            data.building_material = data.building_material.astype('category')
            data.building_type = data.building_type.astype('category')
            data.parking_way = data.parking_way.astype('category')

            city = pd.get_dummies(data.city, drop_first=True)
            city.columns = ['city_' + str(a) for a in city.columns.tolist()]
            data = pd.concat([data, city], axis=1)
            data.drop(columns = ['city'], inplace=True)

            town = pd.get_dummies(data.town, drop_first=True)
            town.columns = ['town_' + str(a) for a in town.columns.tolist()]
            data = pd.concat([data, town], axis=1)
            data.drop(columns = ['town'], inplace=True)

            village = pd.get_dummies(data.village, drop_first=True)
            village.columns = ['village_' + str(a) for a in village.columns.tolist()]
            data = pd.concat([data, village], axis=1)
            data.drop(columns = ['village'], inplace=True)

            building_material = pd.get_dummies(df.building_material, drop_first=True)
            building_material.columns = ['building_material_' + str(a) for a in building_material.columns.tolist()]
            data = pd.concat([data, building_material], axis=1)
            data.drop(columns = ['building_material'], inplace=True)

            building_type = pd.get_dummies(data.building_type, drop_first=True)
            building_type.columns = ['building_type_' + str(a) for a in building_type.columns.tolist()]
            data = pd.concat([data, building_type], axis=1)
            data.drop(columns = ['building_type'], inplace=True)

            parking_way = pd.get_dummies(data.parking_way, drop_first=True)
            parking_way.columns = ['parking_way_' + str(a) for a in parking_way.columns.tolist()]
            data = pd.concat([data, parking_way], axis=1)
            data.drop(columns = ['parking_way'], inplace=True)

            self.data = data
        except:
            print('Error: category')
        
    def correlation(self, dataset, threshold):
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
                    if colname in dataset.columns:
                        del dataset[colname] # deleting the column from the dataset

        return dataset
        
    def extend_feature(self):
        try:
            data = self.data
            test = data[data.log_total_price.isnull()]
            train = data[~data.log_total_price.isnull()]
            tmp = data[['doc_num','master_num','bachelor_num','highschool_num','junior_num','elementary_num']]
            ttmp = self.numpy_triu1(tmp)
            data = pd.concat([data, ttmp], axis=1)
            tmp = data[['I_10000','II_10000','III_10000','IV_10000','V_10000','VI_10000','VII_10000','VIII_10000','IX_10000','X_10000','XI_10000','XII_10000','XIII_10000','XIV_10000']]
            ttmp = self.numpy_triu1(tmp)
            tttmp = self.numpy_triu1(ttmp)
            corr = pd.concat([self.numpy_triu1(tmp),data.log_total_price], axis=1).T[train.index.tolist()].T.corr().abs()['log_total_price'].sort_values(ascending=False)
            ttmp_df = self.correlation(ttmp.copy(), 0.8)
            tttmp_df = self.correlation(tttmp.copy(), 0.8)
            corr = pd.concat([tttmp_df,data.log_total_price], axis=1).T[train.index.tolist()].T.corr().abs()['log_total_price'].sort_values(ascending=False)
            li = corr[corr > 0.4].index.tolist()
            li.remove('log_total_price')
            tttmp_df= tttmp_df[li]
            data = pd.concat([data, ttmp_df, tttmp_df], axis=1)

            tmp = data[['I_5000','II_5000','III_5000','IV_5000','V_5000','VI_5000','VII_5000','VIII_5000','IX_5000','X_5000','XI_5000','XII_5000','XIII_5000','XIV_5000']]
            ttmp = self.numpy_triu1(tmp)
            tttmp = self.numpy_triu1(ttmp)
            ttmp_df = self.correlation(ttmp.copy(), 0.8)
            tttmp_df = self.correlation(tttmp.copy(), 0.8)
            corr = pd.concat([tttmp_df,data.log_total_price], axis=1).T[train.index.tolist()].T.corr().abs()['log_total_price'].sort_values(ascending=False)
            li = corr[corr > 0.4].index.tolist()
            li.remove('log_total_price')
            tttmp_df= tttmp_df[li]
            data = pd.concat([data, ttmp_df, tttmp_df], axis=1)

            tmp = data[['I_1000','II_1000','III_1000','IV_1000','V_1000','VI_1000','VII_1000','VIII_1000','IX_1000','X_1000','XI_1000','XII_1000','XIII_1000','XIV_1000']]
            ttmp = self.numpy_triu1(tmp)
            tttmp = self.numpy_triu1(ttmp)
            ttmp_df = self.correlation(ttmp.copy(), 0.8)
            tttmp_df = self.correlation(tttmp.copy(), 0.8)
            corr = pd.concat([tttmp_df,data.log_total_price], axis=1).T[train.index.tolist()].T.corr().abs()['log_total_price'].sort_values(ascending=False)
            li = corr[corr > 0.4].index.tolist()
            li.remove('log_total_price')
            tttmp_df= tttmp_df[li]
            data = pd.concat([data, ttmp_df, tttmp_df], axis=1)

            self.data = data
        except:
            print('Error: extend_feature')
    def export(self):
        try:
            data = self.data
            test = data[data.log_total_price.isnull()]
            train = data[~data.log_total_price.isnull()]
            print(test.head(5))
            print(train.head(5))
    #         train.to_csv(train1.csv)
    #         test.to_csv(test1.csv)
        except:
            print('Error: export')
    def run(self):
        if not self.load_data():
            print('Error')
        if not self.feature_transform():
            print('Error')
        if not self.category():
            print('Error')
        if not self.extend_feature():
            print('Error')
        if not self.export():
            print('Error')
        
if __name__ == '__main__':
    cf = config()
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        du = data_process(cf, path)
        result = du.run()
#         if result:
#             print(du.Result)
#         else:
#             du.show_log()
        
        