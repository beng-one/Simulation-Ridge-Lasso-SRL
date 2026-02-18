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
    random_state_seed : noyau pour la réproductibilité des données simulées (-->int)
    '''

    # typage des paramètres de la fonction
    if not isinstance(PopulationSize, int):
        raise TypeError("Le type de 'PopulationSize' n'est pas le bon. Il doit être de type 'int'")
    if not isinstance(LoiMoyenne, float):
        raise TypeError("Le type de 'LoiMoyenne' n'est pas le bon. Il doit être de type 'float'")
    if not isinstance(LoiVariance, float):
        raise TypeError("Le type de 'LoiVariance' n'est pas le bon. Il doit être de type 'float'")
    if not isinstance(Random_State_Seed, int):
        raise TypeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'")

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
    Residus : Données des residus (-->np.ndarray)
    '''

    if not isinstance(Residus, np.ndarray):
        raise TypeError("Le type de 'Residus' n'est pas le bon. Il doit être de type 'np.ndarray'")


    Nombre = Residus.shape[0]
    Moyenne = round(Residus.mean(),3)
    Variance = round(Residus.var(),3)
    Q25 = round(np.quantile(Residus, 0.25),3)
    Q50 = round(np.quantile(Residus, 0.50),3)
    Q75 = round(np.quantile(Residus, 0.75),3)
    Min = round(Residus.min(),3)
    Max = round(Residus.max(),3)

    StatisticsValues = [Nombre, Moyenne, Variance, Q25, Q50, Q75, Min, Max]
    StatisticsNames = ['Nombre', 'Moyenne', 'Variance', 'Quantile25', 'Quantile50', 'Quantile75', 'Min', 'Max']

    Dataframe = pd.DataFrame({
        'Noms': StatisticsNames,
        'Valeurs': StatisticsValues,
        })

    return Dataframe



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

    if not isinstance(Residus, np.ndarray):
        raise TypeError("Le type de 'Residus' n'est pas le bon. Il doit être de type 'np.ndarray'")
    if not isinstance(Figure, str):
        raise TypeError("Le type de 'Figure' n'est pas le bon. Il doit être de type 'str'")



    # Statistiques
    Nombre = Residus.shape[0]
    Moyenne = round(Residus.mean(),3)
    Var = round(Residus.var(),3)

    # Histogramme
    if Figure == 'Histogramme':
        sns.histplot(Residus, kde=True)
        xmin, xmax, ymin, ymax = plt.axis()
        plt.title(f"Histogramme des residus simulés")
        plt.text(xmax * (1-0.05), ymax * (1-0.05),f"N={Nombre}\n $\\mu$={Moyenne} \n $\\sigma^{2}$={Var}", ha='right', va='top')
        return plt.plot()

    Q25 = round(np.quantile(Residus, 0.25),3)
    Q50 = round(np.quantile(Residus, 0.50),3)
    Q75 = round(np.quantile(Residus, 0.75),3)

    # Boxplot
    if Figure == 'Boxplot':
        sns.boxplot(Residus)
        plt.title(f"Boxplot des residus simulés")
        return plt.plot()

    else:
        raise ValueError(f"La valeurs associée à l'argument args: Figure est incorrecte")


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
        raise TypeError("Le type de 'PopulationSize' n'est pas le bon. Il doit être de type 'int'")

    if not isinstance(FeatureNumber, int):
        raise TypeError("Le type de 'FeatureNumber' n'est pas le bon. Il doit être de type 'int'")

    if not isinstance(Coefs_B, list):
        raise TypeError("Le type de 'Coefs_B' n'est pas le bon. Il doit être de type 'list'. ")

    if not isinstance(Residus, np.ndarray):
        raise TypeError("Le type de 'Residus' n'est pas le bon. Il doit être de type 'np.ndarray'. ")

    if not isinstance(Random_State_Seed, int):
        raise TypeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'. ")

    if PopulationSize != len(Residus):
        raise ValueError("'PopulationSize' et 'Residus' n'ont pas la même taille. ")
    if FeatureNumber != len(Coefs_B):
        raise ValueError("'FeatureNumber' et 'Coefs_B' n'ont pas la même taille. ")
    if not isinstance(SaveOptionPath, str):
        raise TypeError("Le type de 'SaveOptionPath' n'est pas le bon. Il doit être de type 'str'")

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
        raise TypeError("Le type de 'dataframe_population' n'est pas le bon. Il doit être de type 'pd.DataFrame'")

    if not isinstance(SampleSize, int):
        raise TypeError("Le type de 'SampleSize' n'est pas le bon. Il doit être de type 'int'")

    if not isinstance(SaveOptionPath, str):
        raise TypeError("Le type de 'SaveOptionPath' n'est pas le bon. Il doit être de type 'str'")

    if not isinstance(Random_State_Seed, int):
        raise TypeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'")

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
        raise TypeError("Le type de 'Predictors' n'est pas le bon. Il doit être de type 'pd.DataFrame'.")
    if not isinstance(Target, np.ndarray):
        raise TypeError("Le type de 'Target' n'est pas le bon. Il doit être de type 'np.ndarray.")
    if not isinstance(Alpha_list, np.ndarray):
        raise TypeError("Le type de 'Alpha_list' n'est pas le bon. Il doit être de type 'np.ndarray'.")
    if not isinstance(Intercept, bool):
        raise TypeError("Le type de 'Intercept' n'est pas le bon. Il doit être de type 'bool'.")
    if not isinstance(Random_State_Seed, int):
        raise TypeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'. ")

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
        alpha_elem_name = f"{alpha_elem}"
        summary_coef[alpha_elem_name] = result_complete

    return summary_coef

# Fonction pour ajuster le modèle de régression régularisée LASSO en fonction des valeurs de alpha
def regularized_regressions(Predictors, Target, Alpha_list, Intercept, Model, Random_State_Seed):

    '''
    Objectif:
    ---------
    Fonction pour modéliser y à partir X d'un modèle de régression régularisée en fonction de alpha et de l'intercept.

    Arguments:
    ----------
    Predictors : Variables d'entrée (-->pd.dataframe)
    Target : Variable cible (-->pd.dataframe)
    Alpha_list : Coefficients de pénalisation (-->list)
    Intercept : Constante du modèle (-->bool)
    model : Modèle de régression pénalisée : Ridge, Lasso (-->str)
    Random_State_Seed : noyau pour la réproductibilité des données simulées (-->float or None)

    '''

    # Importation de la librairie Ridge de sklearn
    import sklearn
    from sklearn.linear_model import Ridge, Lasso

    # Vérification des types des arguments
    if not isinstance(Predictors, pd.DataFrame):
        raise TypeError("Le type de 'Predictors' n'est pas le bon. Il doit être de type 'pd.DataFrame'.")
    if not isinstance(Target, np.ndarray):
        raise TypeError("Le type de 'Target' n'est pas le bon. Il doit être de type 'np.ndarray.")
    if not isinstance(Alpha_list, np.ndarray):
        raise TypeError("Le type de 'Alpha_list' n'est pas le bon. Il doit être de type 'np.ndarray'.")
    if not isinstance(Intercept, bool):
        raise TypeError("Le type de 'Intercept' n'est pas le bon. Il doit être de type 'bool'.")
    if not isinstance(Random_State_Seed, int):
        raise TypeError("Le type de 'Random_State_Seed' n'est pas le bon. Il doit être de type 'int'. ")
    if not isinstance(Model, str):
        raise TypeError("Le type de 'Model' n'est pas le bon. Il doit être de type 'str'. ")

    # Création du dataframe
    Predictors_names = list(Predictors.columns)
    Predictors_names.insert(0,'Intercept')
    summary_coef = pd.DataFrame({'Variables': Predictors_names})

    # Modèle de régression Ridge en fonction de alpha
    for alpha_elem in Alpha_list:

        if Model == 'ridge':
            # Modèle de Régression Régularisée Ridge
            penalized_model = Ridge(alpha=alpha_elem, fit_intercept=Intercept, random_state=Random_State_Seed)
        elif Model == 'lasso':
            # Modèle de Régression Régularisée Lasso
            penalized_model = Lasso(alpha=alpha_elem, fit_intercept=Intercept, random_state=Random_State_Seed)
        else:
            raise ValueError(f"Le modèle {Model} sélectionné n'est pas supporté dans la fonction. Les modèles pris en charge sont 'ridge' et 'lasso'.")

        # Résultats du modèles : Intercept et coefficients estimés
        penalized_model.fit(Predictors, Target)
        result_intercept = penalized_model.intercept_
        result_coefficients = penalized_model.coef_
        result_complete = np.insert(result_coefficients, 0, result_intercept)

        # Mise à jour du dataframe
        alpha_elem_name = f"{alpha_elem}"
        summary_coef[alpha_elem_name] = result_complete

    return summary_coef

# Fonction pour visualiser le retrecissement des coefficients en fonction de lambda
def visualization_shrinking(Summary_Coefficients_Lambda):
    '''

    Objectif:
    ---------
    Fonction pour visualisation le retricement des coefficients des régression en fonction de lambda, le paramètre de pénalisation.

    Arguments:
    ----------
    Summary_Coefficients_Lambda : Tableau de synthèse des coeffiients de régression et des paramètres de pénalisation (-->pd.dataframe)

    '''
    if not isinstance(Summary_Coefficients_Lambda, pd.DataFrame):
        raise TypeError("Le type de 'Summary_Coefficients_Lambda' n'est pas le bon. Il doit être de type 'pd.DataFrame'.")

    # Manipulation de la base de données
    df = pd.DataFrame(Summary_Coefficients_Lambda)
    df.set_index(keys='Variables', drop=True, inplace=True)
    df_T = df.T

    # Sélection de la palette de couleurs
    cmaps = plt.colormaps
    colors = cmaps.get_cmap('tab10').resampled(df_T.shape[1]).colors

    # Visualisation
    fig, axs = plt.subplots()  # Automatisatio de figsize
    for i, predictor in enumerate(df_T.columns):
        axs.plot(df_T.index, df_T[predictor], label=predictor, linestyle="-", alpha=1, linewidth=2, marker="*",
                 color=colors[i])  # Personalisation des couleurs

    xmin, xmax, ymin, ymax = plt.axis()
    plt.axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
    plt.title('Evolution des coefficients en fonction de Lambda')
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    plt.grid(True, alpha=0.25, linestyle='--')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

