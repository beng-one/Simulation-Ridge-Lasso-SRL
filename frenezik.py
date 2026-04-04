# IMPORTATION DES LIBRAIRIES --------------------------

# Manipulation des données
import math
import numpy as np
import pandas as pd
import scipy
from numpy.random.mtrand import RandomState

# Visualitation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import seaborn as sns

# Modèles de régression
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from hyperframe.frame import DataFrame
import cvxopt

# Répertoire de travail et gestion d'erreur
import os

# Gestion des erreurs et typage statistique
import warnings
import typing
import mypy

# Configuration de la visualisation
fill_violet_color = "#3d0261"
edge_orange_color = "#fbba6d"

# Fonction pour calculer les residus d'un modèle
def rss_computing(beta, X, y, intercept):

    '''
    Objectif:
    ----------
    Fonction pour calculer la somme des carrés des résidus.

    Arguments:
    ---------
    beta : vecteur associé aux coefficients de régression --> (list, np.ndarray).
    X : matrice associée aux variables explicatives du modèle --> (pd.DataFrame).
    y : vecteur associé à la variable cible --> (list, np.ndarray).
    intercept : intercept du modèle --> (float).

    Sortie:
    -------
    rss : valeur associée à la somme des carrés des résidus --> (float).
    '''

    # calcul de la somme des carrés des résidus
    y_chapo = X @ beta + intercept * np.ones(shape=(X.shape[0], 1))
    rss = np.sum((y - y_chapo) ** 2)

    # sortie du modèle
    return rss

class Residuals():

    '''
    Objectif:
    ----------
    Class permettant de simuler des résidus selon une loi à priori et d'effectuer des analyses descriptives

    Arguments:
    ---------
    population_size : taille de la population souhaitée --> (int).
    random_seed : noyau de reproductibilité --> (float, None).

    Fonctions:
    ---------
    Residuals.Simulation : Fonction pour simuler les résidus de la population selon une loi à priori.
    Residuals.Statistics : Fonction pour calculer les statistiques descriptives des résidus simulés.
    Residuals.Visualization : Fonction pour visualiser la distribution et la dispersion des résidus.
    '''

    def __init__(self, population_size, random_seed):

        # vérification de <<population_size>>
        if not instance(population_size, int):
            raise TypeError("Le type de <<population_size>> n'est pas correct. <<population_size>> doit être de type 'int'.")
        if not population_size >= 0:
            raise ValueError(" La valeur de <<population_size>> n'est pas correcte. <<population_size>> doit être superieur ou égale à 0.")

        # vérification de <<random_seed>>
        if not instance(self.random_seed, (float, None)):
            raise TypeError("Le type de <<random_seed>> n'est pas correct. <<random_seed>> doit être de type 'float' ou 'None.")

        self.population_size = population_size
        self.random_seed = random_seed

    # Fonction pour simuler les résidus selon une loi à priori
    def Simulation(self, law="normal", law_parameters=[0,1]):

        '''
        Objectif:
        ----------
        Fonction pour simuler les résidus de la population selon une loi à priori.

        Arguments:
        ---------
        law : loi suivie par les résidus --> (str).
        law_parameters : paramètres de la loi suivie par les résidus --> (list, np.ndarray).

        Sortie:
        -------
        residuals_simulated : résidus simulés --> (np.ndarray)
        '''

        # vérification de <<law>>
        if not isinstance(law, str):
            raise ValueError("Le type de <<law>> n'est pas correct. <<law>> doit être de type <<str>>.")
        if not law in ("normal", "student", "pareto", "weibull"):
            raise ValueError("La valeur de <<law>> n'est pas correcte. <<law>> doit contenir une valeur parmi : normal, student, pareto, weibull.")
        if not isinstance(law_parameters, (list, np.ndarray)):
            raise TypeError("Le type de <<law_parameters>> n'est pas correct. <<law_parameters>> doit être de type list ou np.ndarray.")

        if law == "normal" and len(law_parameters)!= 2:
            raise ValueError("<<law parameters>> associé à la loi normle doit être de taille 2.")
        if law == "student" and len(law_parameters)!= 1:
            raise ValueError("<<law parameters>> associé à la loi de Student doit être de taille 1.")
        if law == "pareto" and len(law_parameters)!= 1:
            raise ValueError("<<law parameters>> associé à la loi de Pareto doit être de taille 1.")
        if law == "weibull" and len(law_parameters)!= 1:
            raise ValueError("<<law parameters>> associé à la loi de Weibull doit être de taille 1.")

        # Noyau de reproductibilité
        np.random.seed(self.random_seed)

        # Simulation des résidus selon une loi à priori
        if law == "normal":
            residuals_simulated = np.random.normal(loc=law_parameters[0], scale=law_parameters[1], size=self.population_size)
        elif law == "student":
            residuals_simulated = np.random.standard_t(df=law_parameters[0], size=self.population_size)
        elif law == "pareto":
            residuals_simulated = np.random.pareto(a=law_parameters[0], size=self.population_size) # forme de la distribution (a>=0)
        elif law == "weibull":
            residuals_simulated = np.random.weibull(a=law_parameters[0], size=self.population_size)

        # sortie des résidus simulés
        return residuals_simulated

    # Fonction pour afficher les statistiques descriptives des résidus simulés
    def Statistics(self, residuals_simulated, digits):

        '''
        Objectif:
        ----------
        Fonction pour calculer les statistiques descriptives des résidus simulés.

        Arguments:
        ---------
        residuals_simulated : résidus simulés --> (list, np.ndarray).
        digits : nombre de chiffres significatifs après la virgule --> (int).

        Sortie:
        -------
        summary_residuals_statistics : synthèse des statistiques descriptives des résidus simulés  --> (pd.DataFrame).
        '''

        # Vérification de <<residuals_simulated>>
        if not isinstance(residuals_simulated, np.ndarray):
            if isinstance(residuals_simulated, list):
                residuals_simulated = np.array(residuals_simulated)
            else:
                raise TypeError(
                    " Le type de <<residuals_simulated>> n'est pas correct. <<residuals_simulated>> doit être de type list ou np.ndarray.")

        # Vérification de <<digits>>
        if not isinstance(digits, int):
            raise ("Le type de <<digits>> n'est pas correct. <<residuals_simulated>> doit être de type <<int>>")
        if not digits >= 0:
            raise TypeError("La valeur de <<digits>> n'est pas correcte. <<digits>> doit être supérieur ou égal à 0.")

        # Calculs des statistiques descriptives
        self._Count_ = int(residuals_simulated.shape[0])
        self._Mean_ = round(residuals_simulated.mean(), digits)
        self._Variance_ = round(residuals_simulated.var(), digits)
        self._Q25_ = round(np.quantile(residuals_simulated, q=0.25), digits)
        self._Q50_ = round(np.quantile(residuals_simulated, q=0.50), digits)
        self._Q75_ = round(np.quantile(residuals_simulated, q=0.75), digits)
        self._Min_ = round(residuals_simulated.min(), digits)
        self._Max_ = round(residuals_simulated.max(), digits)
        self._Skewness_ = round(scipy.stats.skew(residuals_simulated), digits)
        self._Kurtosis_ = round(scipy.stats.kurtosis(residuals_simulated), digits)

        # Synthèses des statistiques descriptives au format dataframe
        statistics_values = [self._Count_, self._Mean_, self._Variance_, self._Q25_, self._Q50_, self._Q75_, self._Min_, self._Max_, self._Skewness_, self._Kurtosis_]
        statistics_names = ["Count", "Mean", "Variance", "Q25", "Q50", "Q75", "Min", "Max", "Skewness", "Kurtosis"]
        self.summary_residuals_statistics = pd.DataFrame({
            "Name": statistics_names,
            "Value": statistics_values
        })

        # Sortie du résumé des statistiques descriptives
        return self.summary_residuals_statistics

    def Visualization(self, residuals_simulated, figure):

        '''
        Objectif:
        ----------
        Fonction pour visualiser la distribution et la dispersion des résidus.

        Arguments:
        ---------
        residuals_simulated : résidus simulés --> (list, np.ndarray).
        figure : type de figure souhaitée parmi : histogram, boxplot, qqplot --> (str).

        Sortie:
        -------
        histogram : histogramme des résidus simulés --> (plt)
        boxplot : boîte à moustache des residus simulés --> (plt)
        qqplot : qqplot des residus simulés --> (plt)
        '''

        # Vérification de <<residuals_simulated>>
        if not isinstance(residuals_simulated, np.ndarray):
            if isinstance(residuals_simulated, list):
                residuals_simulated = np.array(residuals_simulated)
            else:
                raise TypeError(
                    " Le type de <<residuals_simulated>> n'est pas correct. <<residuals_simulated>> doit être de type list ou np.ndarray.")

        # Vérification de <<figure>>
        if not isinstance(figure, str):
            raise TypeError("Le type de <<figure>> n'est pas correct. <<figure>> doit être de type str.")

        # Histogramme des résidus simulés
        if figure == "histogram":
            sns.histplot(residuals_simulated, stat="density", color=fill_violet_color, linestyle='-', edgecolor=edge_orange_color, alpha=1)
            sns.kdeplot(residuals_simulated, color=edge_orange_color)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.title(f"Histogramme des residus simulés")
            plt.text(xmax * (1-0.05),
                     ymax * (1-0.05),
                     f"$N$={self._Count_}\n"
                     f" $\\mu$={self._Mean_}\n"
                     f" $\\sigma^{2}$={self._Variance_}\n"
                     f" $skewness$={self._Skewness_}\n"
                     f" $kurtosis$={self._Kurtosis_}",
                     ha='right',
                     va='top')
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            return plt.plot()

        # Boîte à moustache des résidus simulés
        if figure == 'boxplot':
            sns.boxplot(x=residuals_simulated, linecolor=edge_orange_color, color=fill_violet_color, linewidth=1)
            plt.title(f"Boxplot des residus simulés")
            plt.xlabel('residus')
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.plot()

        # QQplot des résidus simulés
        if figure == "qqplot":
            pp = sm.ProbPlot(residuals_simulated, fit=True)
            qq = pp.qqplot(marker='.', markerfacecolor=fill_violet_color, markeredgecolor=fill_violet_color, alpha=1,
                           markersize=12)
            sm.qqline(qq.axes[0], line='45', color=edge_orange_color)
            plt.title("QQplot des residus simulées")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.show()
        else:
            raise TypeError("La valeur de <<figure>> n'est pas correcte. <<figure>> doit contenir une valeur parmi : 'histogram', 'boxplot', 'qqplot'.")

class TargetPredictors():

    '''
    Objectif:
    ----------
    Class permettant de simuler les données d'entrée X et de sortie y de la population et d'effectuer des analyses descriptives

    Arguments:
    ---------
    population_size : taille de la population souhaitée --> (int).
    true_coefficients : coefficients réels associés aux variables explicatives X --> (list, np.ndarray).

    Fonctions:
    ---------
    Residuals.Simulation : Fonction pour simuler les variables X et y de la population.
    Residuals.Sampling : Fonction pour simuler les données de l'échantillon à partir des données de la population.
    Residuals.Visualization_X_y : Fonction pour visualiser les données simulées.
    '''

    def __init__(self, population_size, true_coefficients):

        # Vérification de <<population_size>>
        if not isinstance(population_size, int):
            raise TypeError("Le type de <<population_size>> n'est pas correct. <<population_size>> doit être de type 'int'.")
        if not population_size >= 0:
            raise ValueError("La valeur de <<population_size>> n'est pas correcte. <<population_size>> doit être superieur ou égale à 0.")

        # Vérification de <<true_coefficients>>
        if not isinstance(true_coefficients, np.ndarray):
            if isinstance(true_coefficients, list):
                true_coefficients = np.array(true_coefficients)
            else:
                raise TypeError("Le type de <<true_coefficients>> ")

        self.true_coefficients = true_coefficients
        self.population_size = population_size

    def Simulation(self, residuals_simulated):

        '''
        Objectif:
        ----------
        Fonction pour simuler les variables X et y de la population.

        Arguments:
        ---------
        residuals_simulated : résidus simulés --> (list, np.ndarray).

        Sortie:
        -------
        data_simulated_population : données simulées de la population --> (pd.DataFrame).
        '''

        # Vérification de <<residuals_simulated>>
        if not isinstance(residuals_simulated, np.ndarray):
            if isinstance(residuals_simulated, list):
                residuals_simulated = np.array(residuals_simulated)
            else:
                raise TypeError(
                    " Le type de <<residuals_simulated>> n'est pas correct. <<residuals_simulated>> doit être de type list ou np.ndarray.")

        # Variables associées au nombre de coefficients
        number_coefficient = len(self.true_coefficients)

        # Simulation des variables d'entrée X
        X = np.zeros(shape=(self.population_size, number_coefficient))
        for col in range(0, number_coefficient):
            gaussian_params = np.random.rand(1, 2)
            mu = gaussian_params[:, 0]
            sigma = gaussian_params[:, 1]
            X[:, col] = np.random.normal(mu, sigma, self.population_size)

        # Simulation de la variable cible y
        self.true_coefficients = np.array(self.true_coefficients).reshape(1, number_coefficient)
        y = X @ self.true_coefficients.T + residuals_simulated.reshape(self.population_size, 1)

        # Fusionner les données X et y
        X_y_data = np.hstack((y, X))

        # Convertir en dataframe
        X_y_col_names = ['y']
        for i in range(1, number_coefficient + 1):
            col_name = f"X_{i}"
            X_y_col_names.append(col_name)
        self.data_simulated_population = pd.DataFrame(data=X_y_data, columns=X_y_col_names)

        # Sortie des données simulée de la population
        return self.data_simulated_population

    # Fonction pour simuler les données de l'échantillon à partir des données de la population
    def Sampling(self, population_simulated, sample_size):

        '''
        Objectif:
        ----------
        Fonction pour simuler les données de l'échantillon à partir des données de la population.

        Arguments:
        ---------
        population_simulated : données simulées de la population --> (pd.DataFrame).
        sample_size : taille souhaitée de l'échantillon simulé --> (int).

        Sortie:
        -------
        data_simulated_population : données simulées de la population --> (pd.DataFrame).
        '''

        # Vérification de <<population_simulated>>
        if not isinstance(population_simulated, pd.DataFrame):
            raise TypeError("Le type de <<population_simulated>> n'est pas correct. <<population_simulated>> doit être de type pd.DataFrame.")

        # Vérification de <<sample_size>>
        if not isinstance(sample_size, int):
            raise ("Le type de <<sample_size>> n'est pas correct. <<sample_size>> doit être de type <<int>>")
        if not sample_size >= 0:
            raise TypeError("La valeur de <<sample_size>> n'est pas correcte. <<sample_size>> doit être supérieur ou égal à 0.")

        # Constituer de l'échantillon
        data_simulated_sample = population_simulated.sample(sample_size)

        # Sortie des données simulées de l'échantillon
        return data_simulated_sample

    def Visualization_X_y(self, data_simulated, figure):

        '''
        Objectif:
        ----------
        Fonction pour visualiser les données simulées.

        Arguments:
        ---------
        data_simulated : données simulées  --> (list, np.ndarray).
        figure : type de figure souhaitée parmi : lineplot, scatterplot, correlation_matrix --> (str).

        Sortie:
        -------
        lineplot : courbe d'évolution des variables X et y simulés --> (plt).
        scatterplot : nuage de points entre la variable cible y et les variables d'entrée X --> (plt).
        correlation_matrix : matrice de correlation des données simulées --> (plt).
        '''

        # Vérification de <<data_simulated>>
        if not isinstance(data_simulated, pd.DataFrame):
            raise TypeError("Le type de <<data_simulated>> n'est pas correct. <<data_simulated>> doit être de type <<pd.DataFrame>")

        # Vérification de <<figure>>
        if not isinstance(figure, str):
            raise TypeError("Le type de <<figure>> n'est pas correct. <<figure>> doit être de type str.")

        # Personnalisation de la palette des couleurs
        cmaps = plt.colormaps
        variable_number = data_simulated.shape[1]
        color_index_inf = int(variable_number * (1 - 0.70))
        color_index_sup = int(variable_number * (1 + 0.30))
        colors = cmaps.get_cmap("magma").resampled(color_index_sup).colors

        # Graphique en courbe
        if figure == "lineplot":
            fig, axs = plt.subplots()
            for i, col in enumerate(data_simulated.columns):
                axs.plot(data_simulated[col], label=col, linestyle="-", color=colors[color_index_inf+i], linewidth=1) # personnalisation des couleurs
            plt.title(f"Evolution des données simulées")
            plt.xlabel(f"Nombre d'observation")
            plt.ylabel(f"Valeurs")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        # Nuage de points
        elif figure == "scatterplot":
            fig, axs = plt.subplots()
            for i, var in enumerate(data_simulated.columns):
                axs.plot(data_simulated[var], label=var, linestyle="dotted", color=colors[i], marker="")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        # Matrice de correlation
        elif figure == "correlation_matrix":
            data_simulated_corr = data_simulated.corr()
            sns.heatmap(data_simulated_corr, vmin=-1, vmax=1, cmap='magma_r', annot=True, cbar=True,
                        linewidths=1, annot_kws={'fontsize': 9})
            plt.title("Matrice de correlation")
            plt.show()
        else:
            raise ValueError("<<figure>> doit contenir une valeur parmi : 'lineplot', 'scatterplot', 'correlation_matrix'.")

class Regularization():

    '''
    Objectif:
    ----------
    Class permettant de modéliser y en fonction X à l'aide d'une régression régularisée dépendant du paramètre alpha.

    Arguments:
    ---------
    predictors : matrice associée aux variables prédictives X --> (np.ndarray, pd.DataFrame).
    target : variable cible y --> (list, np.ndarray).
    alpha : facteurs de pénalisation --> (list, np.ndarray).
    intercept : intercept du modèle --> (bool).
    random_seed : noyau de reproductibilité --> (float).

    Fonctions:
    ---------
    Residuals.Penalized_Regression : Fonction pour modéliser y en fonction de X à l'aide d'un modèle de régression régularisée.
    Residuals.Visualization_Shrinking : Fonction pour visualiser le rétrecissement des coefficients en fonction de la force du paramètre alpha.
    Residuals.Predict : Fonction pour calculer y prédite et les residus estimés en fonction des variables prédictives et du paramètre alpha à sélectionner.
    Residuals.Vizualisation_Residuals : Fonction pour visualiser en 3D la relation entre les résidus et les coefficients de régression simulés.
    Residuals.gradient_residual_plot : Fonction pour visualiser en 3D la relation entre les résidus et les variables du modèle.
    '''

    def __init__(self, predictors, target, alpha, intercept, random_seed):

        # Vérification de <<predictors>>
        if not isinstance(predictors, pd.DataFrame):
            raise TypeError("Le type de <<predictors>> n'est pas correct. <<predictors>> doit être de type pd.DataFrame.")

        # Vérification de <<target>>
        if not isinstance(target, np.ndarray):
            if isinstance(target, list):
                target = np.array(target)
            else:
                raise TypeError("Le type de <<target>> n'est pas correct. <<target>> doit être de type np.ndarray ou list.")

        # Vérification de <<alpha>>
        if not isinstance(alpha, np.ndarray):
            if isinstance(alpha, list):
                alpha = np.array(alpha)
            else:
                raise TypeError("Le type de <<alpha>> n'est pas correct. <<alpha>> doit être de type np.ndarray ou list.")

        # Vérifiation de <<intercept>>
        if not isinstance(intercept, bool):
            raise TypeError("Le type de <<intercept>> n'est pas correct. <<intercept>> doit être de type bool. ")

        # vérification de <<random_seed>>
        if not instance(self.random_seed, (float, None)):
            raise TypeError("Le type de <<random_seed>> n'est pas correct. <<random_seed>> doit être de type 'float' ou 'None.")

        self.predictors = predictors
        self.target = target
        self.alpha = alpha
        self.intercept = intercept
        self.random_seed = random_seed

    # Fonction pour développer un modèle de régression régularisée
    def Penalized_Regression(self, model):

        '''
        Objectif:
        ----------
        Fonction pour modéliser y en fonction de X à l'aide d'un modèle de régression régularisée.

        Arguments:
        ---------
        model : type de modèle régularisé (Lasso, Ridge)  --> (str).

        Sortie:
        -------
        penalized_coefficients_table : résumé des coefficients pénalisés en fonction de alpha --> (pd.DataFrame).
        '''

        # Vérification de <<model>>
        if not isinstance(model, str):
            raise TypeError("Le type de <<model>> n'est pas correct. <<model>> doit être de type str")

        # Création du dataframe
        predictors_names = list(self.predictors.columns)

        # Intégration de l'intercept dans le vecteur des coefficients de régression
        if self.intercept == True:
            predictors_names.insert(0, "Intercept")
        elif self.intercept == False:
            predictors_names = predictors_names
        else:
            raise ValueError("La valeur associée à l'argument args: intercept est incorrecte.")

        penalized_coefficients_table = pd.DataFrame({
            'Variables':predictors_names,
        })

        # Modèle de régression régularisée
        for alpha_elem in self.alpha:
            if model == 'ridge':
                # Modèle de Régression Régularisée Ridge
                penalized_model = Ridge(alpha=alpha_elem, fit_intercept=self.intercept, random_state=self.random_seed)
            elif model == 'lasso':
                # Modèle de Régression Régularisée Lasso
                penalized_model = Lasso(alpha=alpha_elem, fit_intercept=self.intercept, random_state=self.random_seed)
            else:
                raise ValueError(f"Le modèle {model} sélectionné n'est pas supporté dans la fonction. Les modèles pris en charge sont 'ridge' et 'lasso'.")

            # Résultats du modèles : Intercept et coefficients estimés
            penalized_model.fit(self.predictors, self.target)
            result_intercept = penalized_model.intercept_
            result_coefficients = penalized_model.coef_

            # Intégration de l'intercept dans le vecteur des coefficients de régression
            if self.intercept == True:
                result_complete = np.insert(result_coefficients, 0, result_intercept)
            elif self.intercept == False:
                result_complete = result_coefficients
            else:
                raise ValueError(" L<<intercept>> doit contenir une valeur parmi : 'intercept', 'scatterplot', 'correlation_matrix'.")

            # Mise à jour du dataframe
            alpha_elem_name = f"{alpha_elem}"
            penalized_coefficients_table[alpha_elem_name] = result_complete

            # Sortie du dataframe
        return penalized_coefficients_table


    def Visualization_Shrinking(self, penalized_coefficients_table, variables_selected, alpha_value_selected, figure):

        '''
        Objectif:
        ----------
        Fonction pour visualiser le rétrecissement des coefficients en fonction de la force du paramètre alpha.

        Arguments:
        ---------
        penalized_coefficients_table : résumé des coefficients pénalisés en fonction de alpha --> (pd.DataFrame).
        variables_selected : variables à sélectionner pour la visualisation en 2D --> (list, np.ndarray).
        alpha_value_selected : valeur de alpha à sélectionnée pour la visualisation en 2D --> (int, str).
        figure : type de figure souhaitée parmi : curve et ellipse --> (str).

        Sortie:
        -------
        curve : courbes pour illustrer l'effet de rétrécissement --> (plt).
        ellipse : ellipse pour illustrer l'effet de rétrécissement --> (plt).
        '''

        # Vérification de <<penalized_coefficients_table>>
        if not isinstance(penalizedype_coefficients_table, pd.DataFrame):
            raise TypeError("Le type de <<penalizedype_coefficients_table>> n'est pas correct. <<penalizedype_coefficients_table>> doit être pd.DataFrame.")

        # Vérification de <<variables_selected>>
        if not isinstance(variables_selected, list):
            raise TypeError("Le type de <<variables_selected>> n'est pas correct. <<variables_selected>> doit être list.")

        # Vérification de <<alpha_value_selected>>
        if not isinstance(alpha_value_selected, float):
            raise TypeError("Le type de <<alpha_value_selected>> n'est pas correct. <<alpha_value_selected>> doit être de float.")

        # Vérification de <<figure>>
        if not isinstance(figure, str):
            raise TypeError("Le type de <<figure>> n'est pas correct. <<figure>> doit être de str.")

        # Visualisation du rétrecissement selon une approche par courbe
        if figure == "curve":
            variables_selected == None
            alpha_value_selected == None

            # Manipulation de la base de données
            df = pd.DataFrame(penalized_coefficients_table)
            df.set_index(keys="Variables", drop=True, inplace=True)
            df_T = df.T

            # Sélection de la palette de couleurs
            variable_number = df_T.shape[1]
            color_index_inf = int(variable_number * (1 - 0.70))
            color_index_sup = int(variable_number * (1 + 0.30))
            cmaps = plt.colormaps
            colors = cmaps.get_cmap("magma").resampled(color_index_sup).colors

            # Visualisation
            fig, axs = plt.subplots()  # Automatisation de figsize
            for i, predictor in enumerate(df_T.columns):
                axs.plot(df_T.index, df_T[predictor], label=predictor, linestyle="-", alpha=1, linewidth=2, marker="*",
                         color=colors[color_index_inf+i])  # Personalisation des couleurs
            xmin, xmax, ymin, ymax = plt.axis()
            plt.axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
            plt.title('Evolution des coefficients en fonction de Lambda')
            plt.xlabel("Lambda")
            plt.ylabel("Coefficients")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        # Visualisation du rétrecissement selon une approche par ellipse
        elif figure == "ellipse":
            fig, ax = plt.subplots()

            # Filtre MCO
            variables_filter = penalized_coefficients_table['Variables'].isin(variables_selected)

            # coefficients MCO
            mco_coefficients = penalized_coefficients_table.loc[variables_filter, '0.0'].tolist()

            # coefficients MCO
            penalized_coefficients = penalized_coefficients_table.loc[variables_filter, alpha_value_selected]

            # Ellipses
            angle_params = 125  # 47
            Ellipse3 = Ellipse(xy=(mco_coefficients[0], mco_coefficients[1]), width=9, height=4, angle=angle_params,
                               fill=True, fc='orange', ec='black', alpha=0.50)
            ax.add_patch(Ellipse3)
            Ellipse2 = Ellipse(xy=(mco_coefficients[0], mco_coefficients[1]), width=6, height=3, angle=angle_params,
                               fill=True, fc='orange', ec='black', alpha=0.25)
            ax.add_patch(Ellipse2)
            Ellipse1 = Ellipse(xy=(mco_coefficients[0], mco_coefficients[1]), width=5, height=2, angle=angle_params,
                               fill=True, fc='orange', ec='orange', alpha=1)
            ax.add_patch(Ellipse1)

            # Cercle
            circle_cy = [0, 0]
            radius_value = 2  # np.square(penalized_coefficients[0]**2+penalized_coefficients[1]**2)
            Circle1 = Circle(xy=(circle_cy[0], circle_cy[1]), radius=radius_value, color="lightsteelblue")
            ax.add_patch(Circle1)

            # Configuration de la figure
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])

            ax.annotate("",
                        xy=(0, -10), xycoords='data',
                        xytext=(0, 10), textcoords='data',
                        arrowprops=dict(arrowstyle="<->",
                                        connectionstyle="arc3", color='k', lw=1))

            ax.annotate("",
                        xy=(-10, 0), xycoords='data',
                        xytext=(10, 0), textcoords='data',
                        arrowprops=dict(arrowstyle="<->",
                                        connectionstyle="arc3", color='k', lw=1))

            plt.xlabel(f"{variables_selected[0]}")
            plt.ylabel(f"{variables_selected[1]}")
            plt.plot(mco_coefficients[0], mco_coefficients[1], 'k.')
            plt.text(mco_coefficients[0], mco_coefficients[1], "beta")
            plt.grid(True, alpha=0.15, linestyle='--')
            plt.show()

    # Fonction pour obtenir des valeurs prédites de y et des résidus
    def Predict(self, penalized_coefficients_table, variables_selected, alpha_value, true_target, predictors, purpose):

        '''
        Objectif:
        ----------
        Fonction pour calculer y prédite et les residus estimés en fonction des variables prédictives et du paramètre alpha à sélectionner.

        Arguments:
        ---------
        penalized_coefficients_table : résumé des coefficients pénalisés en fonction de alpha --> (pd.DataFrame).
        variables_selected : variables à sélectionner pour la visualisation en 2D --> (str).
        alpha_value : valeur de alpha à sélectionnée --> (int, float)
        true_target : observations réelles de la variable cible --> (np.ndarray).
        predictors : observations réelles des variable prédictives --> (int, str)
        purpose : type d'opération souhaitée parmi : target_predicted, residuals_predicted, et residuals_sum_square --> (np.ndarray).

        Sortie:
        -------
        target_predicted : valeurs prédites de la variable cible --> (np.ndarray).
        residual_predicted : residus estimés --> (np.ndarray).
        residuals_sum_square : somme des carrés des residus --> (float, int)
        '''

        # Vérificatio de <<penalized_coefficients_table>>
        if not isinstance(penalized_coefficients_table, pd.DataFrame):
            raise TypeError("Le type <<penalized_coefficients_table>> n'est pas correct. <<penalized_coefficients_table>> doit être de type np.ndarray.")

        # Vérification de <<variables_selected>>
        if not isinstance(variables_selected, list):
            raise TypeError("Le type de <<variables_selected>> n'est pas correct. <<variables_selected>> doit être de type list.")

        # Vérification de <<alpha_value>>
        if not isinstance(alpha_value, float):
            raise TypeError("Le type de <<alpha_value>> n'est pas correct. <<alpha_value>> doit être de type float.")

        # Vérification de <<true_target>>
        if not isinstance(true_target, np.ndarray):
            if isinstance(true_target, list):
                true_target = np.array(true_target)
            else:
                raise TypeError("Le type de <<true_target>> n'est pas correct. <<true_target>> doit être de type np.ndarray ou list.")

        # Vérification de <<predictors>>
        if not isinstance(predictors, pd.DataFrame):
            raise TypeError("Le type de <<predictors>> n'est pas correct. <<predictors>> doit être de type pd.DataFrame.")

        if not isinstance(purpose, str):
            raise TypeError("Le type de <<purpose>> n'est pas correct. <<purpose>> doit être de type str.")

        # Vérification de l'intercept dans la table associée au résumé des coefficients pénalisés
        if 'Intercept' in penalized_coefficients_table["Variables"].tolist():
            intercept_filter = penalized_coefficients_table["Variables"].isin(["Intercept"])
            intercept_value = penalized_coefficients_table.loc[intercept_filter, alpha_value][0]
            penalized_coefficients_shape = penalized_coefficients_table.shape[0] - 1

        else:
            intercept_value = 0
            penalized_coefficients_shape = penalized_coefficients_table.shape[0]
        variables_selected_filter = penalized_coefficients_table["Variables"].isin(variables_selected)

        # Choix des coefficients pénalisés en fonction des variables sélectionnées et du paramètre alpha
        penalized_coefficients_filter = penalized_coefficients_table.loc[variables_selected_filter, alpha_value]
        penalized_coefficients_values = penalized_coefficients_filter.tolist()
        penalized_coefficients_index = penalized_coefficients_filter.index.tolist()
        penalized_coefficients_init = np.zeros(penalized_coefficients_shape)

        for i in range(len(penalized_coefficients_index)):
            penalized_coefficients_init[penalized_coefficients_index[i]] = penalized_coefficients_values[i]
        inputs_array = predictors.values

        # calcul de la somme des résidus au carré
        penalized_coefficients_init_reshape = np.array(penalized_coefficients_init).reshape(1,penalized_coefficients_shape)
        predicted_target = inputs_array @ penalized_coefficients_init_reshape.T + intercept_value * np.ones(
            shape=(predictors.shape[0], 1))

        # choix de la sortie
        if purpose == "target_predicted":
            return predicted_target
        elif purpose == "residuals_predicted":
            residuals = (true_target - predicted_target)**2
            return residuals
        elif purpose == "residuals_sum_square":
            residuals = (true_target - predicted_target)**2
            rss = np.sum(residuals)
            return rss
        else:
            raise ValueError("La valeur de <<purpose>> n'est pas correcte. <<purpose>> doit prendre une valeur parmi : 'target_predicted', 'residuals_predicted', 'residuals_sum_square'.")

    # Calcul de l'écart-type, des statistiques, des pvaleurs
    # Evolution des coefficients
    # Evolution des écarts-types

    # Fonction pour visualiser la relation entre les residus du modèle et les coefficients de régression
    def Vizualisation_Residuals(self, list_beta_1, list_beta_2, predictors, target, penalized_coefficients_table, var_1, var_2, figure):

        '''
         Objectif:
         ----------
         Fonction pour visualiser en 3D la relation entre les résidus et les coefficients de régression simulés.

         Arguments:
         ---------
         list_beta_1 : coefficients de régression simulés de la variable 1 --> (list, np.ndarray).
         list_beta_2 : coefficients de régression simulés de la variable 2 --> (list, np.ndarray).
         predictors : variables explicatives du modèle --> (int, float)
         target : variable cible du modèle --> (list, np.ndarray).
         var_1 : variable à sélectionner  pour la visualiser 3D sur l'axe des abscisses --> (str)
         var_2 : variable à sélectionner  pour la visualiser 3D sur l'axe des abscisses  --> (str)
         figure : type de figure souhaitée parmi les figures suivantes : contour_map, surface3D --> (str).

         Sortie:
         -------
         contour_map : courbes de niveau pour visualiser en 3D la relation entre les résidus et les coefficients de régression simulés --> (plt).
         surface3D :Surfaece en 3D pour visualiser la relation entre les résidus et les coefficients de régression simulés --> (plt).
         '''

        # Vérification de <<list_beta_1>>
        if not isinstance(list_beta_1, np.ndarray):
            if isinstance(list_beta_1, list):
                list_beta_1 = np.array(list_beta_1)
            else:
                raise TypeError("Le type de <<list_beta_1>> n'est pas correct. <<list_beta_1>> doit être de type np.ndarray ou list.")

        # Vérification de <<list_beta_2>>
        if not isinstance(list_beta_2, np.ndarray):
            if isinstance(list_beta_2, list):
                list_beta_2 = np.array(list_beta_2)
            else:
                raise TypeError("Le type de <<list_beta_2>> n'est pas correct. <<list_beta_2>> doit être de type np.ndarray ou list.")

        # Vérification de <<predictors>>
        if not isinstance(predictors, pd.DataFrame):
            raise TypeError("Le type de <<predictors>> n'est pas correct. <<predictors>> doit être de type pd.DataFrame.")

        # Vérification de <<true_target>>
        if not isinstance(target, np.ndarray):
            if isinstance(target, list):
                target = np.array(target)
            else:
                raise TypeError("Le type de <<target>> n'est pas correct. <<target>> doit être de type np.ndarray ou list.")

        # Vérificatio de <<penalized_coefficients_table>>
        if not isinstance(penalized_coefficients_table, pd.DataFrame):
            raise TypeError("Le type <<penalized_coefficients_table>> n'est pas correct. <<penalized_coefficients_table>> doit être de type np.ndarray.")

        # Vérification de <<var_1>>
        if not isinstance(var_1, str):
            raise TypeError("Le type de <<var_1>> n'est pas correct. <<var_1>> doit être de type str.")

        # Vérification de <<var_1>>
        if not isinstance(var_2, str):
            raise TypeError("Le type de <<var_2>> n'est pas correct. <<var_2>> doit être de type str.")

        if not {var_1, var_2}.issubset(set(predictors.columns.tolist())):
            raise ValueError(f"La valeurs de <<var_1>> et/ ou la valeur de <<var_2>> n'est pas correcte. <<var_1>> et <<var_2>> doivent prendre des valeurs parmi : {predictors.columns.tolist()}.")

        # Utilisation de la fonction rss_computing pour calculer la somme des carrés des résidus
        global rss_computing

        # Configuration des variables employées
        X = predictors[[var_1, var_2]]
        X_values = X.values
        y_true = target
        b1 = list_beta_1
        b2 = list_beta_2
        rss = np.zeros(shape=(len(list_beta_1), len(list_beta_2)))

        # Calculs de la somme des carrés des résidus
        j = 0
        for p in b1:
            i = 0
            for q in b2:
                rss[i, j] = rss_computing(beta=[p, q], X=X_values, y=y_true, intercept=5)
                i += 1
            j += 1
        B1, B2 = np.meshgrid(b1, b2)

        # Courbe de niveaux
        if figure == "contour_map":
            plt.contour(B1, B2, rss, levels=50, linewidths=0.5, colors='yellow')
            cp = plt.contourf(B1, B2, rss, levels=100, cmap='magma')
            plt.colorbar(cp, label="RSS")
            for enum, col in enumerate(penalized_coefficients_table.columns[1:]):
                list = penalized_coefficients_table.iloc[1:, enum + 1].values
                if col == 'true_coefficients':
                    plt.scatter(list[0], list[1], alpha=1, linestyle='--', color='green', label="true_coefs")
                elif col == "0.0":
                    plt.scatter(list[0], list[1], alpha=1, linestyle='--', color='yellow', label="mco_coefs")
                else:
                    plt.scatter(list[0], list[1], alpha=1, linestyle='--', color='blue', label="penalized_coefs")
            plt.xlabel(f"$\\beta_{var_1[-1]}$")
            plt.ylabel(f"$\\beta_{var_2[-1]}$")
            plt.title("Carte de niveau du RSS")
            plt.grid(True, alpha=0.15)
            plt.show()

        # Surface 3D
        elif figure == "surface3D":
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_rotate_label(True)
            ax.view_init(30, 35)
            ax.plot_surface(B1, B2, rss, cmap="magma", alpha=0.7, antialiased=False)
            ax.plot_wireframe(B1, B2, rss, color='red', alpha=0.1)
            ax.set_xlabel(f"$\\beta_{var_1[-1]}$")
            ax.set_ylabel(f"$\\beta_{var_2[-1]}$")
            ax.set_zlabel("$RSS$", rotation=90)
            ax.set_title("Visualisation 3D de la fonction de perte en fontion des coefficients pénalisés", size=15)

        else:
            raise ValueError("La valeur de <<figure>> n'est pas correcte. <<figure>> doit prendre des valeurs parmi : contour_map, surface3D.")


# Fonction pour visualiser la relation entre les variables explicatives et les résidus simulés
    def gradient_residual_plot(self, data_simulated_coefficients, var_1, var_2, residual, model):

        '''
         Objectif:
         ----------
         Fonction pour visualiser en 3D la relation entre les résidus et les variables du modèle.

         Arguments:
         ---------
         data_simulated_coefficients : coefficients de régression simulés de la variable 1 --> (list, np.ndarray).
         var_1 : variable à sélectionner  pour la visualiser 3D sur l'axe des abscisses --> (str)
         var_2 : variable à sélectionner  pour la visualiser 3D sur l'axe des abscisses  --> (str)
         residual : résidus simulés du modèle --> (str).
         model : modèle de régression employé --> (str).

         Sortie:
         -------
         contour_map : courbes de niveau pour visualiser en 3D la relation entre les résidus et les coefficients de régression simulés --> (plt).
         surface3D :Surfaece en 3D pour visualiser la relation entre les résidus et les coefficients de régression simulés --> (plt).
         '''

        # Vérification de <<data_simulated_coefficients>>
        if not isinstance(data_simulated_coefficients, pd.DataFrame):
            raise TypeError("Le type de <<data_simulated_coefficients>> n'est pas correct. <<data_simulated_coefficients>> doit être de type pd.DataFrame.")

        # Vérification de <<var_1>>
        if not isinstance(var_1, str):
            raise TypeError("Le type de <<var_1>> n'est pas correct. <<var_1>> doit être de type str.")

        # Vérification de <<var_1>>
        if not isinstance(var_2, str):
            raise TypeError("Le type de <<var_2>> n'est pas correct. <<var_2>> doit être de type str.")

        # Vérification de <<var_1>>
        if not isinstance(residual, str):
            raise TypeError("Le type de <<residual>> n'est pas correct. <<residual>> doit être de type str.")

        if not {var_1, var_2, residual}.issubset(set(data_simulated_coefficients.columns.tolist())):
            raise ValueError(
                f"<<var_1>>, <<var_2>> et <<residual>> doivent prendre des valeurs parmi : {data_simulated_coefficients.columns.tolist()}.")

        from scipy.interpolate import griddata
        import matplotlib.tri as tri

        # Données
        df = pd.DataFrame(data_simulated_coefficients)
        X = df[var_1].values
        Y = df[var_2].values
        Z = df[residual].values

        # Coordonnées minimum
        X_min = min(X)
        Y_min = min(Y)
        Z_min = min(Z)

        # Coordonnées Maximum
        X_max = max(X)
        Y_max = max(Y)
        Z_max = max(Z)

        # Création d'une grille régulière
        xi = np.linspace(X_min, X_max, 100)
        yi = np.linspace(Y_min, Y_max, 100)
        Xi, Yi = np.meshgrid(xi, yi)

        # Coefficient de régression régularisée optimale
        coef_opt_ridge = df.loc[df[residual] == Z_min, :]

        # Interpolation
        triang = tri.Triangulation(X, Y)
        interpolar = tri.LinearTriInterpolator(triang, Z)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = interpolar(Xi, Yi)

        # Contour + gradient
        plt.contour(Xi, Yi, Zi, levels=50, linewidths=0.5, colors='yellow')
        cp = plt.contourf(Xi, Yi, Zi, levels=100, cmap='magma')
        plt.colorbar(cp, label="Residual")
        plt.xlabel(f"B_{var_1[-1]}")
        plt.ylabel(f"B_{var_2[-1]}")
        plt.title("Carte de niveau du RSS")
        plt.grid(True, alpha=0.15)
        # plt.legend()
        plt.show()