# %% [markdown]
# ## Implementing Gradient boost model

# %% [markdown]
# Taken from https://square.github.io/pysurvival/models/random_survival_forest.html
# 
# https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html
# 
# https://notebook.community/sebp/scikit-survival/examples/evaluating-survival-models
# 
# https://stats.stackexchange.com/questions/570172/bootstrap-optimism-corrected-results-interpretation

# %%
#pip install pandas
#pip install numpy
#pip install matplotlib
#pip install sklearn
#pip install sksurv
#pip install scikit-learn==0.20.0
#pip install sklearn
#pip install scikit-survival
#pip install optuna
#pip install lifelines

# %%
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import joblib 
import lifelines

# %%
from sklearn.model_selection import train_test_split
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import set_config

set_config(display="text")  # displays text representation of estimators

# %%
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

# %%
from sklearn.utils import resample
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.calibration import calibration_curve
from scipy import stats

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sklearn.model_selection import cross_val_score

from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.utils import resample

from lifelines import KaplanMeierFitter



# %%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from scipy.interpolate import interp1d
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score


# Parse data-dir and output-dir arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, help="Path to directory containing data")
parser.add_argument("--output-dir", type=Path, help="Path to directory to write analysis outputs to")
args = parser.parse_args()

# %%
subset_df = pd.read_csv(args.data_dir / 'synthetic_data_24062024.csv')

# %%
X_train = subset_df.drop(columns = ['X_mi_m','event_cmp','tstop'])

# %%
#X_train

# %%
y = subset_df[['event_cmp','tstop']].to_numpy()
#y

# %% [markdown]
# The fit method expects the y data to be a structured array. In our case, this is an array of Tuples, where the first element is the status and second one is the survival in days.

# %%
aux = [(e1,e2) for e1,e2 in y]

y_train = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
#y_train

# %% [markdown]
# ## Set up and fit random survival forest

# %%
random_state = 20


# %% [markdown]
# # Tuning hyperparameters and variable selection

# %% [markdown]
# ## Hyperparameter tuning

# %%
def objective(trial):
    #loss = trial.suggest_categorical('loss', ['coxph', 'squared'])
    learning_rate = trial.suggest_float('learning_rate', 0.1,1.0)
    n_estimators = trial.suggest_int('n_estimators', 2,100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
    max_depth = trial.suggest_int('max_depth', 1, 100)
    max_features = trial.suggest_int('max_features', 1,100)    
    dropout_rate  = trial.suggest_float('dropout_rate', 0, 1.0)
    subsample = trial.suggest_float('subsample', 0, 1.0) 
    

    boost_tree  = GradientBoostingSurvivalAnalysis(
        #loss = loss,
        learning_rate = learning_rate,
        n_estimators = n_estimators,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_depth = max_depth,
        max_features = max_features,    
        dropout_rate  = dropout_rate,
        subsample = subsample
    )
    
    #feature_selection = SelectKBest(score_func=f_classif, k=40)
    #PCA not appropriate as data variables have a clear clinical meaning
    rf = Pipeline([#('scl', StandardScaler()),
                      #('pca', PCA(n_components=10)),
                      #('feature_selection', feature_selection), 
                      ('boost_tree', boost_tree)
    ])

    rf.fit(X_train, y_train)
    
    scores = cross_val_score(estimator=rf, 
                         X=X_train, 
                         y=y_train, 
                         cv=10,
                         n_jobs=1)

    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, gc_after_trial = True, show_progress_bar = True)

# %%
print('Best trial:')
trial = study.best_trial
print('  Value: {:.5f}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# %%
boost_tree = GradientBoostingSurvivalAnalysis(
        #loss = trial.params['loss'],
        learning_rate = trial.params['learning_rate'],
        n_estimators = trial.params['n_estimators'],
        min_samples_split = trial.params['min_samples_split'],
        min_samples_leaf = trial.params['min_samples_leaf'],
        max_depth = trial.params['max_depth'],
        max_features = trial.params['max_features'],    
        dropout_rate  = trial.params['dropout_rate'],
        subsample = trial.params['subsample']
    )
#feature_selection = SelectKBest(score_func=f_classif, k=45)
rf = Pipeline([#('scl', StandardScaler()),
                     # ('pca', PCA(n_components=10)), 
                     #('feature_selection', feature_selection),
                      ('boost_tree', boost_tree)
    ])

# %%

rf.fit(X_train, y_train)

# %%
# save model with joblib 
filename = args.output_dir / 'joblib_boost_model_20062024.sav'
joblib.dump(rf, filename)

# %%
#import joblib 
#rf = joblib.load('//live.rd.ucl.ac.uk/ritd-ag-project-rd01fz-rgidd68/test/joblib_boost_model_0811_per_COX.sav')

# %%
print('C-score')
print(rf.score(X_train, y_train))

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=rf, 
                         X=X_train, 
                         y=y_train, 
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores from cross validation: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# %% [markdown]
# ## Measuring performance - initially on training data

# %% [markdown]
# ## AUC

# %% [markdown]
# Apparant AUC using predictions made by the model developed on the full original dataset

# %%
##Alternative approach to AUC using predict.survival.function instead (will be better for ensemble model)
surv_times = np.arange(0.25, 5.25, 0.25)
survival_functions = rf.predict_survival_function(X_train)
survival_probabilities = np.array([sf(surv_times) for sf in survival_functions])

rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
    y_train, y_train, -survival_probabilities, surv_times
)


#plt.plot(surv_times, rsf_auc, marker="o")
#plt.axhline(rsf_mean_auc, linestyle="--")
#plt.xlabel("years from enrollment")
#plt.ylabel("time-dependent AUC")
#plt.grid(True)

# %%
print('Survival times plotted')
print(surv_times)
print('AUC values')
print(rsf_auc)
print('Mean AUC over all time points')
print(rsf_mean_auc)

# %%
#X_train = X_train.drop(columns = ['X_mi_m'])

# %%
#CI for apparant AUC
#use if restrict to 5 years follow up time


n_bootstraps = 1000
auc_array_CI = []
surv_times = np.arange(0.25, 5.25, 0.25)

for i in range(n_bootstraps):
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
 
    # make predictions using the survival forest model
    survival_functions = rf.predict_survival_function(X_boot)
    survival_probabilities = np.array([sf(surv_times) for sf in survival_functions])
     
    #AUC
    
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
        y_train, y_boot, -survival_probabilities, surv_times
    )
        
    auc_array_CI.append(rsf_auc)
    
# Calculate 95% confidence interval using the percentile method
lower_bounds = [[] for _ in range(len(surv_times))]
upper_bounds = [[] for _ in range(len(surv_times))]

for j in range(len(surv_times)):
    lower_bound = np.percentile([auc[j] for auc in auc_array_CI], 2.5)
    upper_bound = np.percentile([auc[j] for auc in auc_array_CI], 97.5)
    lower_bounds[j].append(lower_bound)
    upper_bounds[j].append(upper_bound)

combined_array = np.array(auc_array_CI)
average = np.mean(combined_array, axis = 0)

# Print the 95% confidence intervals for each time point
for j in range(len(surv_times)):
    print(f'AUC Time Point {surv_times[j]}: Mean: {average[j]:.4f} 95% Confidence Interval: ({np.mean(lower_bounds[j]):.4f}, {np.mean(upper_bounds[j]):.4f})')

# %% [markdown]
# Internal validation using bootstrap samples = Optimism corrected AUC
# Need to bootstrap for each imputed dataset

# %%
#use if restrict follow up time to 5 years


n_bootstraps = 1000

auc_array = []
auc_array_orig = []
optimism_auc=[]

surv_times = np.arange(0.25, 5.25, 0.25)

for i in range(n_bootstraps):
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
    
    # Fit the best parameter model, on the bootstrapped data
    boot_fit = rf.fit(X_boot, y_boot)
    
    # make predictions using the survival forest model
    survival_functions = boot_fit.predict_survival_function(X_boot)
    survival_probabilities = np.array([sf(surv_times) for sf in survival_functions])

    #AUC
    
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
        y_boot, y_boot,-survival_probabilities, surv_times
    )
    
    auc_array.append(rsf_auc)
    
    # 2. Get statistics on the original dataset
        # make predictions using the survival forest model
    survival_functions = boot_fit.predict_survival_function(X_train)
    survival_probabilities_orig = np.array([sf(surv_times) for sf in survival_functions])
       
 
    #AUC
    rsf_auc_orig, rsf_mean_auc_orig = cumulative_dynamic_auc(
        y_boot, y_train, -survival_probabilities_orig , surv_times
    )
        
    auc_array_orig.append(rsf_auc_orig)
        
        
    #3. calculate optimism
    auc_diff = np.subtract(rsf_auc, rsf_auc_orig)
    optimism_auc.append(auc_diff)
    

# %%
combined_array = np.array(optimism_auc)
average = np.mean(combined_array, axis = 0)
print('optimism for AUC values')
print(surv_times )
print(average)

# %%
# Columns that correspond to the one-hot encoded "Race" categories
Tstage_columns = ['Tstage=T1a',
    'Tstage=T1b',
    'Tstage=T2a',
    'Tstage=T2b',
    'Tstage=T3',
    'Tstage=T4']
# Function to reverse the one-hot encoding
def reverse_one_hot_encoding(df, base_category, columns):
    # Create a new column 'Race' with the base category as default
    df['Tstage'] = base_category
    
    # Assign the corresponding category based on the one-hot encoded columns
    for column in columns:
        category = column.split('=')[1]
        df.loc[df[column] == 1, 'Tstage'] = category
    
    return df


# %%

# Reverse one-hot encoding for the "Race" variable
df_Tstage = X_train.copy()
df_Tstage = reverse_one_hot_encoding(df_Tstage, 'T0', Tstage_columns)


# %%
# Check counts of each category in the new 'Race' column
print("Unique values in 'Tstage' column:", df_Tstage['Tstage'].unique())

print("Value counts in 'Tstage' column:")
print(df_Tstage['Tstage'].value_counts())

print(df_Tstage['Tstage=T1a'].value_counts())
print(df_Tstage['Tstage=T1b'].value_counts())
print(df_Tstage['Tstage=T2a'].value_counts())
print(df_Tstage['Tstage=T2b'].value_counts())
print(df_Tstage['Tstage=T3'].value_counts())
print(df_Tstage['Tstage=T4'].value_counts())

# %%
surv_times = np.arange(1, 4.25, 0.25)

unique_Tstages = ['T1a',
    'T1b',
    'T2a',
    'T2b',
    'T3',
    'T4','T0']

# %%
# Use if utilising only the one imputed data set
auc_by_Tstage= {}


for Tstage in unique_Tstages:
    index = (df_Tstage['Tstage'] == Tstage)
    X_train_Tstage = df_Tstage.loc[index]
    y_train_Tstage = y_train[index]
    X_train_Tstage = X_train_Tstage.drop(columns = ['Tstage'])
    #index = (X_train[f"Race={ethnicity}"] == 1)
    rsf_risk_scores = rf.predict(X_train_Tstage)
    rsf_auc_Tstage, rsf_mean_auc = cumulative_dynamic_auc(
        y_train_Tstage, y_train_Tstage, rsf_risk_scores, surv_times
    )
    auc_by_Tstage[Tstage] = rsf_auc_Tstage

    
# Display AUC values for each ethnicity
for Tstage, auc in auc_by_Tstage.items():
    print(f'AUC for {Tstage}: {auc}')
    print(surv_times)


# %% [markdown]
# ## E/O ratio

# %%
survival_functions_test = rf.predict_survival_function(X_train)

# %%
#Get observed events using kaplein meier

custom_time_points = np.arange(0.25, 5.25, 0.25)

kmf = KaplanMeierFitter()
kmf.fit(durations = y_train['Survival_in_days'], event_observed = y_train['Status'], timeline =custom_time_points )



# %%
#expected probability at time t
desired_time = 1
probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
average_probability= np.mean(probabilities_at_desired_time)
EO_ratio = (1-kmf.survival_function_at_times(desired_time))/(1-average_probability)
print('EO_ratio at time 1 year')
print(EO_ratio)

# %%
desired_time = 2
probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
average_probability= np.mean(probabilities_at_desired_time)
EO_ratio = (1-kmf.survival_function_at_times(desired_time))/(1-average_probability)
print('EO_ratio at time 2 year')
print(EO_ratio)

# %%
desired_time = 5
probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
average_probability= np.mean(probabilities_at_desired_time)
EO_ratio = (1-kmf.survival_function_at_times(desired_time))/(1-average_probability)
print('EO_ratio at time 5 year')
print(EO_ratio)

# %%
#CI for E/O
#use if restrict to 5 years follow up time

n_bootstraps = 1000
eo_ratios_bootstrap_CI = []

# Specify your custom time points
desired_time = 1


for i in range(n_bootstraps):
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
 
    # make predictions using the survival forest model
    survival_functions_test = rf.predict_survival_function(X_boot)
    
    # Calculate the expected number of events at custom time points
    custom_time_points = np.arange(0.25, 5.25, 0.25)

    kmf = KaplanMeierFitter()
    kmf.fit(durations = y_boot['Survival_in_days'], event_observed = y_boot['Status'], timeline =custom_time_points )
    expected_events_t = kmf.survival_function_at_times(desired_time)
    
    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
    observed_events_t= np.mean(probabilities_at_desired_time)
    
    EO_ratio = (1-expected_events_t)/(1-observed_events_t)
    eo_ratios_bootstrap_CI.append(EO_ratio)
    


# %%
confidence_interval = np.percentile(eo_ratios_bootstrap_CI, [2.5, 97.5]) 
print('Confidence interval at time one year')
print(confidence_interval)

# %%
#CI for E/O
#use if restrict to 5 years follow up time

n_bootstraps = 1000
eo_ratios_bootstrap2_CI = []

# Specify your custom time points
desired_time = 2


for i in range(n_bootstraps):
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
 
    # make predictions using the survival forest model
    survival_functions_test = rf.predict_survival_function(X_boot)
    
    # Calculate the expected number of events at custom time points
    custom_time_points = np.arange(0.25, 5.25, 0.25)

    kmf = KaplanMeierFitter()
    kmf.fit(durations = y_boot['Survival_in_days'], event_observed = y_boot['Status'], timeline =custom_time_points )
    expected_events_t = kmf.survival_function_at_times(desired_time)
    
    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
    observed_events_t= np.mean(probabilities_at_desired_time)
    
    EO_ratio = (1-expected_events_t)/(1-observed_events_t)
    eo_ratios_bootstrap2_CI.append(EO_ratio)
    


# %%
confidence_interval = np.percentile(eo_ratios_bootstrap2_CI, [2.5, 97.5]) 
print('Confidence interval at time two years')
print(confidence_interval)

# %%
#CI for E/O
#use if restrict to 5 years follow up time

n_bootstraps = 1000
eo_ratios_bootstrap5_CI = []

# Specify your custom time points
desired_time = 5


for i in range(n_bootstraps):
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
 
    # make predictions using the survival forest model
    survival_functions_test = rf.predict_survival_function(X_boot)
    
    # Calculate the expected number of events at custom time points
    custom_time_points = np.arange(0.25, 5.25, 0.25)

    kmf = KaplanMeierFitter()
    kmf.fit(durations = y_boot['Survival_in_days'], event_observed = y_boot['Status'], timeline =custom_time_points )
    expected_events_t = kmf.survival_function_at_times(desired_time)
    
    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
    observed_events_t= np.mean(probabilities_at_desired_time)
    
    EO_ratio = (1-expected_events_t)/(1-observed_events_t)
    eo_ratios_bootstrap5_CI.append(EO_ratio)
    


# %%
confidence_interval = np.percentile(eo_ratios_bootstrap5_CI, [2.5, 97.5]) 
print('Confidence interval at time five years')
print(confidence_interval)

# %%
#Optimism
n_bootstraps = 1000

eo_ratios_bootstrap = []
eo_ratios_bootstrap_orig = []
optimism_auc=[]

custom_time_points = np.arange(0.25, 5.25, 0.25)
desired_time = 1

for i in range(n_bootstraps):
    
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
    
    # Fit the best parameter model, on the bootstrapped data
    boot_fit = rf.fit(X_boot, y_boot)
    
    # 2. Get statistics on the bootstrap dataset
    # For bootstrap data: make predictions using the survival forest model
    survival_functions_test = boot_fit.predict_survival_function(X_boot)
    # For bootstrap data: Calculate the expected number of events at custom time points 
    kmf = KaplanMeierFitter()
    kmf.fit(durations = y_boot['Survival_in_days'], event_observed = y_boot['Status'], timeline =custom_time_points )
    expected_events_t = kmf.survival_function_at_times(desired_time)

    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
    observed_events_t= np.mean(probabilities_at_desired_time)
    
    EO_ratio = (1-expected_events_t)/(1-observed_events_t)
    eo_ratios_bootstrap.append(EO_ratio)      
   
    # 2. Get statistics on the original dataset
    # For original data: make predictions using the survival forest model
    survival_functions_orig = boot_fit.predict_survival_function(X_train)
        # make predictions using the survival forest model
    kmf_orig = KaplanMeierFitter()
    kmf_orig.fit(durations = y_train['Survival_in_days'], event_observed = y_train['Status'], timeline =custom_time_points )
    expected_events_t_orig = kmf_orig.survival_function_at_times(desired_time)

    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time_orig = [sf(desired_time) for sf in survival_functions_orig]
    observed_events_t_orig= np.mean(probabilities_at_desired_time_orig)
    
    EO_ratio_orig = (1-expected_events_t_orig)/(1-observed_events_t_orig)
    eo_ratios_bootstrap_orig.append(EO_ratio_orig)      
    
        
    #3. calculate optimism
    EO_diff = np.subtract(EO_ratio, EO_ratio_orig)
    optimism_auc.append(EO_diff)
  

# %%
mean_value = np.mean(optimism_auc, axis=0) 
print('Optimism for EO value at time one year')
print(mean_value)

# %%
#Optimism
n_bootstraps = 1000

eo_ratios_bootstrap2 = []
eo_ratios_bootstrap_orig2 = []
optimism_auc2=[]

custom_time_points = np.arange(0.25, 5.25, 0.25)
desired_time = 2

for i in range(n_bootstraps):
    
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
    
    # Fit the best parameter model, on the bootstrapped data
    boot_fit = rf.fit(X_boot, y_boot)
    
    # 2. Get statistics on the bootstrap dataset
    # For bootstrap data: make predictions using the survival forest model
    survival_functions_test = boot_fit.predict_survival_function(X_boot)
    # For bootstrap data: Calculate the expected number of events at custom time points 
    kmf = KaplanMeierFitter()
    kmf.fit(durations = y_boot['Survival_in_days'], event_observed = y_boot['Status'], timeline =custom_time_points )
    expected_events_t = kmf.survival_function_at_times(desired_time)

    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
    observed_events_t= np.mean(probabilities_at_desired_time)
    
    EO_ratio = (1-expected_events_t)/(1-observed_events_t)
    eo_ratios_bootstrap2.append(EO_ratio)      
   
    # 2. Get statistics on the original dataset
    # For original data: make predictions using the survival forest model
    survival_functions_orig = boot_fit.predict_survival_function(X_train)
        # make predictions using the survival forest model
    kmf_orig = KaplanMeierFitter()
    kmf_orig.fit(durations = y_train['Survival_in_days'], event_observed = y_train['Status'], timeline =custom_time_points )
    expected_events_t_orig = kmf_orig.survival_function_at_times(desired_time)

    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time_orig = [sf(desired_time) for sf in survival_functions_orig]
    observed_events_t_orig= np.mean(probabilities_at_desired_time_orig)
    
    EO_ratio_orig = (1-expected_events_t_orig)/(1-observed_events_t_orig)
    eo_ratios_bootstrap_orig2.append(EO_ratio_orig)      
    
        
    #3. calculate optimism
    EO_diff = np.subtract(EO_ratio, EO_ratio_orig)
    optimism_auc2.append(EO_diff)
  

# %%
mean_value = np.mean(optimism_auc2, axis=0) 
print('Optimism for EO value at time two years')
print(mean_value)

# %%
#Optimism

n_bootstraps = 1000

eo_ratios_bootstrap5 = []
eo_ratios_bootstrap_orig5 = []
optimism_auc5=[]

custom_time_points = np.arange(0.25, 5.25, 0.25)
desired_time = 5

for i in range(n_bootstraps):
    
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
    
    # Fit the best parameter model, on the bootstrapped data
    boot_fit = rf.fit(X_boot, y_boot)
    
    # 2. Get statistics on the bootstrap dataset
    # For bootstrap data: make predictions using the survival forest model
    survival_functions_test = boot_fit.predict_survival_function(X_boot)
    # For bootstrap data: Calculate the expected number of events at custom time points 
    kmf = KaplanMeierFitter()
    kmf.fit(durations = y_boot['Survival_in_days'], event_observed = y_boot['Status'], timeline =custom_time_points )
    expected_events_t = kmf.survival_function_at_times(desired_time)

    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time = [sf(desired_time) for sf in survival_functions_test]
    observed_events_t= np.mean(probabilities_at_desired_time)
    
    EO_ratio = (1-expected_events_t)/(1-observed_events_t)
    eo_ratios_bootstrap5.append(EO_ratio)      
   
    # 2. Get statistics on the original dataset
    # For original data: make predictions using the survival forest model
    survival_functions_orig = boot_fit.predict_survival_function(X_train)
        # make predictions using the survival forest model
    kmf_orig = KaplanMeierFitter()
    kmf_orig.fit(durations = y_train['Survival_in_days'], event_observed = y_train['Status'], timeline =custom_time_points )
    expected_events_t_orig = kmf_orig.survival_function_at_times(desired_time)

    # Calculate the observed number of events at custom time points
    probabilities_at_desired_time_orig = [sf(desired_time) for sf in survival_functions_orig]
    observed_events_t_orig= np.mean(probabilities_at_desired_time_orig)
    
    EO_ratio_orig = (1-expected_events_t_orig)/(1-observed_events_t_orig)
    eo_ratios_bootstrap_orig5.append(EO_ratio_orig)      
    
        
    #3. calculate optimism
    EO_diff = np.subtract(EO_ratio, EO_ratio_orig)
    optimism_auc5.append(EO_diff)
  

    

# %%
mean_value = np.mean(optimism_auc5, axis=0) 
print('Optimism for EO value at time five years')
print(mean_value)

# %% [markdown]
# ## Brier score

# %% [markdown]
# A Brier score of 0 means perfect accuracy, and a Brier score of 1 means perfect inaccuracy

# %% [markdown]
# Apparant brier score using predictions made by the model developed on the full original dataset

# %%
unique_values, counts = np.unique(y_train["Survival_in_days"], return_counts=True)


# set time period - use if restrict f/u to 5 years
rsf_times = unique_values[unique_values <=5]


# %% [markdown]
# iterate over the predicted survival functions on the test data and evaluate each at the time points from above.

# %%
rsf_surv_prob = np.row_stack([
    fn(rsf_times)
    for fn in rf.predict_survival_function(X_train)
])


# %% [markdown]
# we want to have a baseline to tell us how much better our models are from random. A random model would simply predict 0.5 every time.

# %%
random_surv_prob = 0.5 * np.ones(
    (y_train.shape[0], rsf_times.shape[0])
)


# %% [markdown]
# kaplein meier estimate also

# %%


km_func = StepFunction(
    *kaplan_meier_estimator(y_train["Status"], y_train["Survival_in_days"])
)
km_surv_prob = np.tile(km_func(rsf_times), (y_train.shape[0], 1))

# %% [markdown]
# use the integrated Brier score (IBS) over all time points, which will give us a single number to compare the models by.

# %%
score_brier = pd.Series(
    [
        integrated_brier_score(y_train, y_train, prob, rsf_times)
        for prob in (rsf_surv_prob, random_surv_prob, km_surv_prob)
    ],
    index=["RSF",  "Random", "Kaplan-Meier"],
    name="IBS"
)
print('Brier score')
print(score_brier)

# %%
#CI for brier score
#use if restrict to 5 years follow up time

n_bootstraps = 1000
score_brier_rsf_CI= []



for i in range(n_bootstraps):
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
    
    #BRIER SCORE
    #Get time span for brier score    
    unique_values, counts = np.unique(y_train["Survival_in_days"], return_counts=True)
    rsf_times = unique_values[unique_values <=5]
    
    #predict for model
    rsf_surv_prob = np.row_stack([
        fn(rsf_times)
        for fn in rf.predict_survival_function(X_boot)
    ])

    
    score_brier = integrated_brier_score(y_train, y_boot, rsf_surv_prob, rsf_times) 
    score_brier_rsf_CI.append(score_brier)

# %%
lower_bound = np.percentile(score_brier_rsf_CI, 2.5)
upper_bound = np.percentile(score_brier_rsf_CI, 97.5)
print('CI for Brier score')
print(lower_bound, upper_bound)

# %% [markdown]
# Internal validation using bootstrap samples = Optimism corrected Brier
# Need to bootstrap for each imputed dataset

# %%
#use if restrict follow up time to 5 years


n_bootstraps = 1000

score_brier_rsf= []

score_brier_rsf_orig= []

optimism_brier = []


for i in range(n_bootstraps):
    # create a bootstrap sample of the test data
    X_boot, y_boot = resample(X_train, y_train, replace=True)
    
    # Fit the best parameter model, on the bootstrapped data
    boot_fit = rf.fit(X_boot, y_boot)
    
    # 1. Get statistics on the bootstrapped predictions
      
    #BRIER SCORE
    #Get time span for brier score    
    unique_values, counts = np.unique(y_boot["Survival_in_days"], return_counts=True)
    rsf_times = unique_values[unique_values <=5]
    
    #predict for model
    rsf_surv_prob = np.row_stack([
        fn(rsf_times)
        for fn in boot_fit.predict_survival_function(X_boot)
    ])

    
    score_brier = integrated_brier_score(y_boot, y_boot, rsf_surv_prob, rsf_times)

    
    score_brier_rsf.append(score_brier)

    
    # 2. Get statistics on the original dataset
    #BRIER SCORE
       
    #predict for model
    rsf_surv_prob_orig = np.row_stack([
        fn(rsf_times)
        for fn in boot_fit.predict_survival_function(X_train)
    ])

    
    score_brier_orig = integrated_brier_score(y_boot, y_train, rsf_surv_prob_orig, rsf_times)
    
    score_brier_rsf_orig.append(score_brier_orig)
    
    # 3. Calcuate optimism
    optimism_brier.append(score_brier - score_brier_orig)

    
            

    

# %%
optimism_array = np.array(optimism_brier)
mean_value = np.mean(optimism_array, axis=0)
std_value = np.std(optimism_array, axis=0)
print('optimism for Brier score (mean, std)')
print(mean_value, std_value)
