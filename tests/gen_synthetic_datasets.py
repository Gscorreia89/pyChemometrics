import pandas as pds
import os
import numpy as np
import sklearn.datasets as datamaker


class_dataset = datamaker.make_classification(n_samples=50, n_features=200, n_informative=25,
                                              n_redundant=5, n_repeated=0, n_classes=2,
                                              n_clusters_per_class=2, weights=None,
                                              flip_y=0.01, class_sep=1.5, hypercube=True,
                                              shift=0.0, scale=1.0, shuffle=True,
                                              random_state=35624)

multiclass_dataset = datamaker.make_classification(n_samples=50, n_features=200,
                                                   n_informative=25, n_redundant=5,
                                                   n_repeated=0, n_classes=4,
                                                   n_clusters_per_class=2, weights=None,
                                                   flip_y=0.01, class_sep=1.5, hypercube=True,
                                                   shift=0.0, scale=1.0, shuffle=True,
                                                   random_state=35624)

regression_dataset = datamaker.make_regression(n_samples=50, n_features=200,
                                               n_informative=25, n_targets=1, bias=0.0, effective_rank=None,
                                               tail_strength=0.5, noise=0.0, shuffle=True,
                                               coef=False, random_state=35624)

regression_block_dataset = datamaker.make_regression(n_samples=50, n_features=200,
                                                     n_informative=25, n_targets=5,
                                                     bias=0.0, effective_rank=None,
                                                     tail_strength=0.5, noise=0.0,
                                                     shuffle=True, coef=False,
                                                     random_state=35624)
# Save the 2 class classification dataset
col_names = ['Class']
col_names.extend(['Var_' + x for x in map(str, range(class_dataset[0].shape[1]))])
class_dataset_frame = pds.DataFrame(data=np.c_[class_dataset[1], class_dataset[0]], columns=col_names)
save_path = os.path.dirname(os.path.abspath("__file__"))
class_dataset_frame.to_csv(os.path.join(save_path, './test_data/classification_twoclass.csv'), index=False)

# Save the multiclass classification dataset
multiclass_dummy_mat = pds.get_dummies(multiclass_dataset[1]).values
col_names = ['Class_Vector']
col_names.extend(['Class_' + x for x in map(str, range(np.atleast_2d(multiclass_dummy_mat).shape[1]))])
col_names.extend(['Var_' + x for x in map(str, range(multiclass_dataset[0].shape[1]))])
multiclass_dataset_frame = pds.DataFrame(data=np.c_[multiclass_dataset[1], multiclass_dummy_mat,
                                                    multiclass_dataset[0]], columns=col_names)
save_path = os.path.dirname(os.path.abspath("__file__"))
multiclass_dataset_frame.to_csv(os.path.join(save_path, './test_data/classification_multiclass.csv'), index=False)

# Save the 1 Y variable regression dataset
col_names = ['Y']
col_names.extend(['Var_' + x for x in map(str, range(regression_dataset[0].shape[1]))])
regression_dataset_frame = pds.DataFrame(data=np.c_[regression_dataset[1], regression_dataset[0]],
                                         columns=col_names)
save_path = os.path.dirname(os.path.abspath("__file__"))
regression_dataset_frame.to_csv(os.path.join(save_path, './test_data/regression.csv'), index=False)

# Save the multi Y regression dataset
col_names = (['Y_' + x for x in map(str, range(np.atleast_2d(regression_block_dataset[1]).shape[1]))])
col_names.extend(['Var_' + x for x in map(str, range(regression_block_dataset[0].shape[1]))])
regression_block_dataframe = pds.DataFrame(data=np.c_[regression_block_dataset[1], multiclass_dataset[0]],
                                           columns=col_names)
save_path = os.path.dirname(os.path.abspath("__file__"))
regression_block_dataframe.to_csv(os.path.join(save_path, './test_data/regression_multiblock.csv'), index=False)
