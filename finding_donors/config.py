__author__ = "rgr"

# data
TRAINING_DATA_FILE = 'data/census.csv'
PIPELINE_NAME = 'finding_donors_pipeline'

TARGET = 'income'

# input variables
FEATURES = ['age', 'workclass', 'education_level', 'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# variables to log transform
NUMERICAL_LOG_VARS = ['capital-gain', 'capital-loss']

# categorical variables to encode
CATEGORITCAL_VARS = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'native-country']
