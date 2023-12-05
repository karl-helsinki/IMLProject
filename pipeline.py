# Import required packages
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

data_dir = Path("data")
train_data_path = data_dir / "gecko_ml.csv"
test_data_path = data_dir / "gecko_ml_test.csv"

def prepare_data(path: Path, features: list, target: list=None):
    """Load data from the given path, select required features, preprocess the data
    and if needed select the target
    """
    dataset = pd.read_csv(path)
    if not target:
        # Test dataset lacks this column, but it is present in the training set
        dataset["ohe_parentspecies_decane_toluene"] = 0.0
    
    x = dataset[features]
    
    # One hot encoding of parentspecies column has Nan values, 
    # I decided to keep the rows with Nans and set each of the ohe* variables to one 
    x[x[["ohe_parentspecies_apin",
    "ohe_parentspecies_apin_decane",
    "ohe_parentspecies_apin_decane_toluene",
    "ohe_parentspecies_apin_toluene",
    "ohe_parentspecies_decane",
    "ohe_parentspecies_decane_toluene"]].isna()] = 1.0

    # id column is not a meanigful feature
    x = x.drop("id", axis=1)
    if target:
        y = dataset[target]
        return x, y
    else:
        return x
    
def transform_predictions(predictions):
    """Reverse transformation applied to the target variable. First unscales the variable based on the mean and std values from the training set
    then applies exp function. Verified to work properly on the train set."""
    df = pd.read_csv("data/train.csv")
    orig_log_mean = np.log(df["pSat_Pa"]).mean()
    orig_log_std = np.log(df["pSat_Pa"]).std()
    unscaled = unscale(predictions, orig_log_mean, orig_log_std)
    return np.exp(unscaled)

def unscale_orig_target(target):
    """Unscale the value of pSat_Pa to obtain the original values. Verified to work properly for the train set."""
    df = pd.read_csv("data/train.csv")
    orig_mean = df["pSat_Pa"].mean()
    orig_std = df["pSat_Pa"].std()
    return unscale(target, orig_mean, orig_std)

def unscale(variable, mean, std):
    return std * variable + mean

def scale(variable, mean, std):
    return (variable - mean) / std

def save_test_predictions(test_predictions, save_path, original_test_file):
    original_test = pd.read_csv(original_test_file)
    original_test["target"] = test_predictions
    original_test["Id"] = original_test["id"]
    result = original_test[["Id", "target"]]
    result.to_csv(save_path, index=False)

def cross_validate_model(model, x, y, cv=5, scoring='r2'):
    scores = cross_val_score(model, x, y, cv=cv, scoring=scoring)
    return scores.mean()

def find_best_feature_subset(coeff, model, x, y, start=0):
    abs_coeff = np.abs(coeff)
    search_space = np.sort(abs_coeff)[start:-1][::-1]
    search_space = search_space[search_space != 0] # don't use features with zero coefficents
    scores = []
    feats = []
    for i, val in enumerate(search_space):
        selected_feats = x.columns[abs_coeff >= val]
        if selected_feats.size == 0:
            continue
        scores.append(cross_validate_model(model, x[selected_feats], y, cv=5))
        feats.append(selected_feats.values)
        print(f"Num features: {selected_feats.size}, current score: {scores[-1]}")
    max_score = np.max(scores)
    best_feats = feats[np.argmax(scores)]
    return best_feats, max_score, feats, scores

def validate_models(models: dict, x, y):
    result = {}
    for name, model in models.items():
        print(f"Validating model {name}")
        score = cross_validate_model(model, x, y)
        result[name] = score
    return result

def load_x_y(features, target="log10"):
    dependant = ['pSat_Pa', 'trans_pSat_Pa']
    x, y = prepare_data(train_data_path, features, target=dependant)
    x, y = shuffle(x, y)
    y_org = y[dependant[0]]
    y_trans = y[dependant[1]]
    y_log10 = np.log10(unscale_orig_target(y_org))
    y_log10_scaled = scale(y_log10, y_log10.mean(), y_log10.std())
    if target == "original":
        return x, y_org
    elif target == "transformed":
        return x, y_trans
    elif target == "log10":
        return x, y_log10
    elif target == "log10_scaled":
        return x, y_log10_scaled
    else:
        raise KeyError
    
def _feature_selection(coef, model, x, y, k):
    best = find_best_feature_subset(coef, model, x, y)
    best_feats, best_feats_score, feats, feats_scores = best
    if k == 1:
        return best_feats, np.std(feats_scores)
    else:
        sortedsargs = np.argsort(feats_scores)[-k:]
        return feats[sortedsargs[::-1]], feats_scores[sortedsargs[::-1]]

def linear_model_feature_selection(models, x, y, k=1):
    models["elasticnet"].fit(x, y)
    return _feature_selection(models["elasticnet"].coef_, models["lin_reg"], x, y, k)

def selectkbest_feature_selection(models, x, y, k=1):
    fs = SelectKBest(f_regression, k='all')
    fs.fit(x, y)
    return _feature_selection(fs.scores_, models["lin_reg"], x, y, k)

def save_feats(feats, filename):
    df = pd.DataFrame([], index=feats)
    df.to_csv(f"features/{filename}")

def find_best_model(models):
    features = ['id', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumOfN',
        'NumHBondDonors', 'NumOfConfUsed', 'cc', 'ccco',
        'hydroxyl_alkl', 'aldehyde', 'ketone', 'carboxylic_acid', 'ester',
        'ether_alicyclic', 'nitrate', 'nitro', 'aromatic_hydroxyl',
        'carbonylperoxynitrate', 'peroxide', 'hydroperoxide',
        'carbonylperoxyacid', 'nitroester', 'trans_NumOfConf',
        'trans_MW',
        'ohe_parentspecies_apin', 'ohe_parentspecies_apin_decane',
        'ohe_parentspecies_apin_decane_toluene',
        'ohe_parentspecies_apin_toluene', 'ohe_parentspecies_decane',
        'ohe_parentspecies_decane_toluene']
    features_full = ['id', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumOfN',
        'NumHBondDonors', 'NumOfConfUsed', 'cc', 'ccco',
        'hydroxyl_alkl', 'aldehyde', 'ketone', 'carboxylic_acid', 'ester',
        'ether_alicyclic', 'nitrate', 'nitro', 'aromatic_hydroxyl',
        'carbonylperoxynitrate', 'peroxide', 'hydroperoxide',
        'carbonylperoxyacid', 'nitroester', 'trans_NumOfConf',
        'trans_MW', 'new_MW_hydroxyl_alkl_interaction',
        'ohe_parentspecies_apin', 'ohe_parentspecies_apin_decane',
        'ohe_parentspecies_apin_decane_toluene', 'new_polarity_score', 'new_num_pca_1', 'new_num_pca_2',
        'ohe_parentspecies_apin_toluene', 'ohe_parentspecies_decane',
        'ohe_parentspecies_decane_toluene']
    x, y = load_x_y(features_full)
    x_test = prepare_data(test_data_path, features)

    lin_feats, lin_std = linear_model_feature_selection(models, x, y)
    skb_feats, skb_std = selectkbest_feature_selection(models, x, y)
    print(f"Lin feats:\n {lin_feats}")
    print(f"SKB feats:\n {skb_feats}")
    print(lin_std, skb_std)
    lin_x = x[lin_feats]
    skb_x = x[skb_feats]
    lin_scores = validate_models(models, lin_x, y)
    skb_scores = validate_models(models, skb_x, y)
    print(lin_scores)
    print(skb_scores)
    save_feats(lin_feats, f"lin_svrscore_{lin_scores['svr']:0.4f}.csv" )
    save_feats(skb_feats, f"skb_svrscore_{skb_scores['svr']:0.4f}.csv" )
    return lin_feats, skb_feats

def generate_test_preds(model, feature_dir='features', pred_dir='predictions'):
    features_full = ['id', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumOfN',
        'NumHBondDonors', 'NumOfConfUsed', 'cc', 'ccco',
        'hydroxyl_alkl', 'aldehyde', 'ketone', 'carboxylic_acid', 'ester',
        'ether_alicyclic', 'nitrate', 'nitro', 'aromatic_hydroxyl',
        'carbonylperoxynitrate', 'peroxide', 'hydroperoxide',
        'carbonylperoxyacid', 'nitroester', 'trans_NumOfConf',
        'trans_MW', 'new_MW_hydroxyl_alkl_interaction',
        'ohe_parentspecies_apin', 'ohe_parentspecies_apin_decane',
        'ohe_parentspecies_apin_decane_toluene', 'new_polarity_score', 'new_num_pca_1', 'new_num_pca_2',
        'ohe_parentspecies_apin_toluene', 'ohe_parentspecies_decane',
        'ohe_parentspecies_decane_toluene']
    x, y = load_x_y(features_full)
    x_test = prepare_data(test_data_path, features_full)
    
    ft_dir = Path(feature_dir)
    for file in ft_dir.glob("*"):
        feats = pd.read_csv(file).iloc[:, 0]
        model.fit(x[feats], y)
        preds = model.predict(x_test[feats])
        save_test_predictions(preds, f"{pred_dir}/{file.name}", test_data_path)

if __name__ == "__main__":
    # Define the models
    models = {}
    models["lin_reg"] = LinearRegression()
    models["rf"] = RandomForestRegressor()
    models["svr"] = SVR()
    models["gbr"] = GradientBoostingRegressor()
    models["elasticnet"] = ElasticNet(alpha=0.003, l1_ratio=1)
    # lin_feats, skb_feats = find_best_model(models)
    generate_test_preds(models["svr"])
    
#     svr = GridSearchCV(
#     SVR(kernel="rbf", gamma=0.1),
#     param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
# )
