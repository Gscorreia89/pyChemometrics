from sklearn.pipeline import make_pipeline
from ..ChemometricsScaler import ChemometricsScaler

from sklearn.model_selection import KFold, GroupKFold, BaseCrossValidator
from sklearn.metrics import accuracy_score, roc_auc_score