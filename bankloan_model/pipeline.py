import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from bankloan_model.config.core import config
from bankloan_model.processing.features import Mapper

xgb_params = {
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 15),
    'min_child_weight': (1, 10),
    'gamma':  (0, 5),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'reg_alpha': (0.0, 50.0),
    'n_estimators': (100, 180, 200, 300),
    'eval_metric': 'auc'
}

def create_pipeline(trial = None):
     if trial:
        model_params = {
            'max_depth': trial.suggest_int('max_depth', *xgb_params['max_depth']),
            'learning_rate': trial.suggest_float('learning_rate', *xgb_params['learning_rate'], log=True),
            'n_estimators': trial.suggest_int('n_estimators', xgb_params['n_estimators'][0], xgb_params['n_estimators'][-1]),
            'min_child_weight': trial.suggest_int('min_child_weight', *xgb_params['min_child_weight']),
            'subsample': trial.suggest_float('subsample', *xgb_params['subsample'])
        }
        xgb_classifier = XGBClassifier(**model_params)
     else:
        xgb_classifier = XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            min_child_weight=1,
            subsample=0.8,
            eval_metric='logloss'
        )

     bankloan_pipe = Pipeline([
     ("map_sex", Mapper(config.model_config_.gender_var, config.model_config_.gender_mappings)
      ),
     ("map_education", Mapper(config.model_config_.education_var, config.model_config_.education_mappings )
     ),
     ("map_home_own", Mapper(config.model_config_.home_own_var, config.model_config_.home_own_mappings)
     ),
     ("map_intent", Mapper(config.model_config_.intent_var, config.model_config_.intent_mappings)
     ),
     ("map_previous_defaults", Mapper(config.model_config_.previous_defaults_var, config.model_config_.previous_defaults_mappings)
     ),
     ('model_xg', xgb_classifier)
     ])
     return bankloan_pipe