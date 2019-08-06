class config:
    def __init__(
        self,
        transform = ['txn_floor','simple_total_floor','txn_dt_year','building_complete_dt_year',
                    'city_population','city_area','city_population_density','log_total_price','doc_num',
                    'master_num','bachelor_num','highschool_num','jobschool_num','junior_num','elementary_num',
                    'born_num','death_num','marriage_num','divorce_num','death/born','divorce/marriage','town_income_median_mean',
                    'town_income_median_min','town_income_median_max','town_income_median_median']):
        self.transforms = transform
        
    def get_transforms(self):
        return self.transforms