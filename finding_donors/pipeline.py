__author__ = "rgr"

import finding_donors.config as config

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import finding_donors.preprocessors as pp

income_pipe = Pipeline(
    [
        ('rare_label_encoder',
         pp.RareLabelCategoricalEncoder(variables=config.CATEGORITCAL_VARS)),

        ('log_transformer',
         pp.LogTransformer(variables=config.NUMERICAL_LOG_VARS))
    ]
)
