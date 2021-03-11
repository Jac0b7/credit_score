import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_rows', None, 'display.width', None)

def analysis(data):
    summary = data.describe().T
    summary['relative'] = 1 - summary['count'] / len(data)
    print(summary)
    print(len(data.dropna(how='any', axis=0)) / len(data))
    return None


# Import Data:
data = pd.read_csv('cs-data.csv', sep=',')
data.drop('ClientId', inplace=True, axis=1)

# Outliers:
print('Remove Outliers')
    # Z-score:
def z_score(df):
    return (df - df.mean())/(df.std(ddof=0))
data.loc[(z_score(data['MonthlyIncm'].copy()) > 3),
         'MonthlyIncm'] = np.NAN
data.loc[(z_score(data['NoOfDependents'].copy()) > 3),
         'NoOfDependents'] = np.NAN
data.loc[(z_score(data['NoRealEstateLoansOrLines'].copy()) > 3),
         'NoRealEstateLoansOrLines'] = np.NAN
data.loc[(z_score(data['NoOfOpenCreditLinesAndLoans'].copy()) > 3),
         'NoOfOpenCreditLinesAndLoans'] = np.NAN
    # Age
data.loc[data['AgeInYears'] < 18, 'AgeInYears'] = np.NAN
    # Limits:
data.loc[data['UtilizationOfUnsecLines'] > 1, 'UtilizationOfUnsecLines'] = np.NAN
    # Income and debt:
data.loc[(data['MonthlyIncm'].isna()), 'DebtRtio'] = np.NAN
    # Being due:
data.loc[data['NoOfTime35-65DaysPastDueNotWorse'] > 80,
         'NoOfTime35-65DaysPastDueNotWorse'] = np.NAN
data.loc[data['NoOfTime60-89DaysPastDueNotWorse'] > 80,
         'NoOfTime60-89DaysPastDueNotWorse'] = np.NAN
data.loc[data['NoOfTimes90DaysLate'] > 80,
         'NoOfTimes90DaysLate'] = np.NAN


# Missing Data:
print('Fill/remove missing Data')
data['missing'] = 0
data.loc[data.MonthlyIncm.isna(), 'missing'] = 1
data['DebtRtio'] = data['DebtRtio'].fillna(data['DebtRtio'].median())
data['MonthlyIncm'] = data['MonthlyIncm'].fillna(data['MonthlyIncm'].median())

data = data.dropna(how='any', axis=0)


# Export:
data.to_pickle('clean_cs-data.pkl')
