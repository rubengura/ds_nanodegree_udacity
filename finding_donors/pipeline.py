__author__ = "rgr"

import finding_donors.config as config

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import finding_donors.preprocessors as pp

income_pipe = Pipeline(
    [
        ('rare_label_encoder',
         pp.RareLabelCategoricalEncoder(tol=0.05, variables=config.CATEGORITCAL_VARS)),

        ('categorical_encoder',
         pp.CategoricalEncoder(variables=config.CATEGORITCAL_VARS)),

        ('log_transformer',
         pp.LogTransformer(variables=config.NUMERICAL_LOG_VARS)),

        ('scaler', MinMaxScaler()),
        ('gbm_classifier', GradientBoostingClassifier(random_state=42))
    ]
)
