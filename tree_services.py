warnings.simplefilter('always', DeprecationWarning)

def _prepare_data_and_categories(X, categories, replacement_for_infrequent, categorical_features_idxs):
    """
        For categories that after OHE have "infrequent" column, we should do
            1) replace in category [1, 3, 'infrequent'] -> [1, 3, non_existing_encoded_value]
            2) replace all values that are not in [1, 3] with non_existing_encoded_value
        non_existing_encoded_value = any numeric value that is does not exist in encoded data, we could use -2
        all this done to have both in categories and X only numeric data

        X - data either as DataFrame or numpy array
        categories[i] - array of categorical values, for example
             [array(['infrequent']),
             array([0, 1]),
             array([1, 2, 'infrequent']
             ]
        replacement_for_infrequent - what numeric value will be used for "infrequent" replacement
        categorical_features_idxs - indexes of categorical features in X (X can include not only categorical features).
            Therefore, len(categorical_features_idxs) = len(categories)"""
    pass

def requires_tree(func):
    pass

def _scores_to_df(scores, only_averages=True):
    pass
