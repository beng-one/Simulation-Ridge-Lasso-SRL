# IMPORTATION DES LIBRAIRIES --------------------------

# Manipulation des données
import math
import numpy as np
import pandas as pd

# Visualitation
import matplotlib.pyplot as plt
import seaborn as sns

# Modèles de régression
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Vérification des résultats de sklearn
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import cvxopt

# Répertoire de travail et gestion d'erreur
import os
import warnings


# FONCTIONS --------------------------

# Fonction pour simuler les residus
def simulation_residus(PopulationSize=1000, LoiMoyenne=0, LoiVariance=1, Random_State_Seed=1980):
    '''
    Objectif:
    ---------
    Fonction pour simuler la distribution gaussienne des résidus au sein de la population

    Arguments:
    ----------
    PopulationSize : Taille de la population (-->int)
    LoiMoyenne : Paramètre associé à la Moyenne de la loi normale (-->float)
    LoiVariance : Paramètre associé à la Variance de la loi normale (-->float)
    random_state_seed : noyau pour la réproductibilité des données simulées (-->float or None)
    '''

    # typage des paramètres de la fonction
    isinstance(PopulationSize, int)
    isinstance(LoiMoyenne, float)
    isinstance(LoiVariance, float)

    np.random.seed(Random_State_Seed)
    residus = np.random.normal(LoiMoyenne, LoiVariance, PopulationSize)
    return residus

# Fonction pour analyser les residus à partir des statistiques descriptivec
def statistics_residus(Residus):


    '''
    Objectif:
    ---------
    Fonction pour analyser les résidus à partir des paramètres de tendance centrale, de dispersion, et de visualisation.

    Arguments:
    ----------
    Residus : Données des residus (-->np.array)
    '''

    if isinstance(Residus, np.ndarray) == True :
        Nombre = round(len(Residus),0)
        Moyenne = Residus.mean()
        Variance = Residus.var()
        Q25 = np.quantile(Residus, 0.25)
        Q50 = np.quantile(Residus, 0.50)
        Q75 = np.quantile(Residus, 0.75)
        Min = Residus.min()
        Max = Residus.max()

        StatisticsValues = [Nombre, Moyenne, Variance, Q25, Q50, Q75, Min, Max]
        StatisticsNames = ['Nombre', 'Moyenne', 'Variance', 'Quantile25', 'Quantile50', 'Quantile75', 'Min', 'Max']

        Dataframe = pd.DataFrame({
            'Noms': StatisticsNames,
            'Valeurs': StatisticsValues,
        })

        return Dataframe

    else:
        print("Residus n'est pas de type np.array")

# Fonction pour analyser les residus à partir des visualisations.
def visualization_residus(Residus, Figure):
    '''
    Objectif:
    ---------
    Fonction pour analyser les résidus à partir des visualisations.

    Arguments:
    ----------
    Residus : Données des residus (-->np.array)
    Figure : Type de figure souhaite
    '''

    if isinstance(Residus, np.ndarray) == True:

        # Statistiques
        Moyenne = Residus.mean()
        Var = Residus.var()
        Nombre = Residus.shape[0]

        # Histogramme
        if Figure == 'Histogramme':
            sns.histplot(Residus, kde=True)
            plt.title(
                f"Histogramme des residus simulés : {Residus.shape[0]} observations, {round(Residus.mean(),2)} Moyenne, {round(Residus.var(),2)} Variance ")
            return plt.plot()

        # Boxplot
        if Figure == 'Boxplot':

            sns.boxplot(Residus)
            plt.title(
                f"Boxplot des residus simulés : {Residus.shape[0]} observations, {round(Residus.mean(),2)} Moyenne, {round(Residus.var(),2)} Variance ")
            return plt.plot()

        else:
            print("La valeurs associée à l'argument args: {figure} est incorrecte")

# Fonction pour simuler les variables d'entrée X et la variable cible y
def simulation_X_y(PopulationSize=1000, FeatureNumber=8, Coefs_B = [1,2,3,4,5,6,7,8], Residus=[1,2,3], SaveOptionPath='', Random_State_Seed=1980):

    '''
    Objectif:
    ----------
    Fonction pour simuler les données d'entrées X et la variable y

    Arguments:
    ----------
    PopulationSize : Taille de la population (-->int)
    FeatureNumber : Nombres de Variables d'entrées souhaitée (-->int)
    Coefs_B : Vecteurs des coefficiens (réels) de la population (-->np.ndarray)
    Residus : vecteur associé aux residus simulés (-->np.ndarray)
    SaveOptionPath : Chemin utilisé pour la sauvegarde de la base de données relative à l'échantillon (-->str)
    Random_State_Seed : noyau pour la réproductibilité des données simulées (-->float or None)
    '''

    # Vérification des types des arguments
    if not isinstance(PopulationSize, int):
        raise AttributeError("Le type de 'PopulationSize' n'est pas le bon. Il doit être de type 'int'")

    if not isinstance(FeatureNumber, int):
        raise AttributeError("Le type de 'FeatureNumber' n'est pas le bon. Il doit être de type 'int'")

    if not isinstance(Coefs_B, list):
        raise AttributeError("Le type de 'Coefs_B' n'est pas le bon. Il doit être de type 'list'. ")

    if not isinstance(Residus, np.ndarray):
        raise AttributeError("Le type de 'Residus' n'est pas le bon. Il doit être de type 'np.ndarray'. ")

    if not isinstance(Random_State_Seed, int):
        raise AttributeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'. ")

    if PopulationSize != len(Residus):
        raise ValueError("'PopulationSize' et 'Residus' n'ont pas la même taille. ")
    if FeatureNumber != len(Coefs_B):
        raise ValueError("'FeatureNumber' et 'Coefs_B' n'ont pas la même taille. ")
    if not isinstance(SaveOptionPath, str):
        raise AttributeError("Le type de 'SaveOptionPath' n'est pas le bon. Il doit être de type 'str'")

    # Simulation des variables d'entrée X
    X = np.zeros(shape=(PopulationSize, FeatureNumber))
    for col in range(0,FeatureNumber):
        gaussian_params = np.random.rand(1,2)
        mu = gaussian_params[:,0]
        sigma = gaussian_params[:,1]
        X[:,col] = np.random.normal(mu, sigma, PopulationSize)

    # Simulation de la variable cible y
    Coefs_B = np.array(Coefs_B).reshape(1,FeatureNumber)
    y =  X @ Coefs_B.T + Residus.reshape(PopulationSize,1)

    # Fusionner les données X et y
    X_y_data = np.hstack((y, X))

    # Convertir en dataframe
    X_y_col_names = ['y']
    for i in range(1,FeatureNumber+1):
        col_name = f"X_{i}"
        X_y_col_names.append(col_name)

    dataframe = pd.DataFrame(data=X_y_data, columns=X_y_col_names)
    dataframe.to_csv(f"{SaveOptionPath}/data_population.csv", index=False)

    return dataframe

# Fonction pour constituer un échantillon à partir des données de la population
def sampling(dataframe_population, SampleSize, SaveOptionPath, Random_State_Seed):
    '''
    Objectif:
    ---------
    Fonction pour constituer l'échantillon et séparer le jeu de données.

    Arguments:
    ----------
    dataframe_population : Données simulées de la population (-->pd.DataFrame)
    SampleSize : Taille de l'échantillon souhaitée (-->int)
    SaveOptionPath : Chemin utilisé pour la sauvegarde de la base de données relative à l'échantillon (-->str)
    Random_State_Seed : noyau pour la réproductibilité des données simulées (-->float or None)
    '''

    if not isinstance(dataframe_population, pd.DataFrame):
        raise AttributeError("Le type de 'dataframe_population' n'est pas le bon. Il doit être de type 'pd.DataFrame'")

    if not isinstance(SampleSize, int):
        raise AttributeError("Le type de 'SampleSize' n'est pas le bon. Il doit être de type 'int'")

    if not isinstance(SaveOptionPath, str):
        raise AttributeError("Le type de 'SaveOptionPath' n'est pas le bon. Il doit être de type 'str'")

    if not isinstance(Random_State_Seed, int):
        raise AttributeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'")

    dataframe_sample = dataframe_population.sample(SampleSize)
    dataframe_sample.to_csv(f"{SaveOptionPath}/data_sample.csv", index=False)

    return dataframe_sample

# Fonction pour ajuster le modèle de régression régularisée de LASSO en fonction des valeurs de alpha
def ridge_finetuning(Predictors, Target, Alpha_list, Intercept, Random_State_Seed):

    '''
    Objectif:
    ---------
    Fonction pour trouver les hyperparamètres de la régression deRidge en fonction de alpha et de l'intercept.

    Arguments:
    ----------
    Predictors : Variables d'entrée (-->pd.dataframe)
    Target : Variable cible (-->pd.dataframe)
    Alpha_list : Coefficients de pénalisation (-->list)
    Intercept : Constante du modèle (-->bool)
    Random_State_Seed : noyau pour la réproductibilité des données simulées (-->float or None)
    '''

    # Importation de la librairie Ridge de sklearn
    import sklearn
    from sklearn.linear_model import Ridge

    # Vérification des types des arguments
    if not isinstance(Predictors, pd.DataFrame):
        raise AttributeError("Le type de 'Predictors' n'est pas le bon. Il doit être de type 'pd.DataFrame'.")
    if not isinstance(Target, np.ndarray):
        raise AttributeError("Le type de 'Target' n'est pas le bon. Il doit être de type 'np.ndarray.")
    if not isinstance(Alpha_list, np.ndarray):
        raise AttributeError("Le type de 'Alpha_list' n'est pas le bon. Il doit être de type 'np.ndarray'.")
    if not isinstance(Intercept, bool):
        raise AttributeError("Le type de 'Intercept' n'est pas le bon. Il doit être de type 'bool'.")
    if not isinstance(Random_State_Seed, int):
        raise AttributeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'. ")

    # Création du dataframe
    Predictors_names = list(Predictors.columns)
    Predictors_names.insert(0, 'Intercept')
    summary_coef = pd.DataFrame({'Variables': Predictors_names})

    # Modèle de régression Ridge en fonction de alpha
    for alpha_elem in Alpha_list:
        # Modèle de Régression Régularisée Ridge
        Ridge_model = Ridge(alpha=alpha_elem, fit_intercept=Intercept, random_state=Random_State_Seed)
        Ridge_model.fit(Predictors, Target)

        # Résultats du modèles : Intercept et coefficients estimés
        result_intercept = Ridge_model.intercept_
        result_coefficients = Ridge_model.coef_
        result_complete = np.insert(result_coefficients, 0, result_intercept)

        # Mise à jour du dataframe
        alpha_elem_name = f"Ridge_Alpha_{alpha_elem}"
        summary_coef[alpha_elem_name] = result_complete

    return summary_coef

# Fonction pour ajuster le modèle de régression régularisée LASSO en fonction des valeurs de alpha
def lasso_finetuning(Predictors, Target, Alpha_list, Intercept, Random_State_Seed):

    '''
    Objectif:
    ---------
    Fonction pour trouver les hyperparamètres de la régression LASSO en fonction de alpha et de l'intercept

    Arguments:
    ----------
    Predictors : Variables d'entrée (-->pd.dataframe)
    Target : Variable cible (-->pd.dataframe)
    Alpha_list : Coefficients de pénalisation (-->list)
    Intercept : Constante du modèle (-->bool)
    Random_State_Seed : noyau pour la réproductibilité des données simulées (-->float or None)
    '''

    # Importation de la librairie Ridge de sklearn
    import sklearn
    from sklearn.linear_model import Lasso

    # Vérification des types des arguments
    if not isinstance(Predictors, pd.DataFrame):
        raise AttributeError("Le type de 'Predictors' n'est pas le bon. Il doit être de type 'pd.DataFrame'.")
    if not isinstance(Target, np.ndarray):
        raise AttributeError("Le type de 'Target' n'est pas le bon. Il doit être de type 'np.ndarray.")
    if not isinstance(Alpha_list, np.ndarray):
        raise AttributeError("Le type de 'Alpha_list' n'est pas le bon. Il doit être de type 'np.ndarray'.")
    if not isinstance(Intercept, bool):
        raise AttributeError("Le type de 'Intercept' n'est pas le bon. Il doit être de type 'bool'.")
    if not isinstance(Random_State_Seed, int):
        raise AttributeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'. ")


    # Création du dataframe
    Predictors_names = list(Predictors.columns)
    Predictors_names.insert(0,'Intercept')
    summary_coef = pd.DataFrame({'Variables': Predictors_names})

    # Modèle de régression Ridge en fonction de alpha
    for alpha_elem in Alpha_list:

        # Modèle de Régression Régularisée Ridge
        Lasso_model = Lasso(alpha=alpha_elem, fit_intercept=Intercept, random_state=Random_State_Seed)
        Lasso_model.fit(Predictors, Target)

        # Résultats du modèles : Intercept et coefficients estimés
        result_intercept = Lasso_model.intercept_
        result_coefficients = Lasso_model.coef_
        result_complete = np.insert(result_coefficients, 0, result_intercept)

        # Mise à jour du dataframe
        alpha_elem_name = f"Ridge_Alpha_{alpha_elem}"
        summary_coef[alpha_elem_name] = result_complete

    return summary_coef