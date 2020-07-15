__author__ = "rgr"

# data
TRAINING_DATA_FILE = 'data/census.csv'
PIPELINE_NAME = 'finding_donors_pipeline'

TARGET = 'income'

# input variables
FEATURES = ['age', 'workclass', 'education_level', 'education_num', 'marital_status', 'occupation', 'relationship',
            'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']

# variables to log transform
NUMERICAL_LOG_VARS = ['capital_gain', 'capital_loss']

# categorical variables to encode
CATEGORITCAL_VARS = ['workclass', 'education_level', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                     'native_country']
