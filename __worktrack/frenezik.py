# IMPORTATION DES LIBRAIRIES --------------------------

# Manipulation des données
import math
import numpy as np
import pandas as pd
import scipy

# Visualitation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import seaborn as sns

# Modèles de régression
import sklearn
from hyperframe.frame import DataFrame
from numpy.random.mtrand import RandomState
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import cvxopt

# Répertoire de travail et gestion d'erreur
import os

# Gestion des erreurs et typage statistique
import warnings
import typing
import mypy

# Fonction pour calculer les residus d'un modèle
def rss_computing(beta, X, y, intercept):
    y_chapo = X @ beta + intercept * np.ones(shape=(X.shape[0], 1))
    rss = np.sum((y - y_chapo) ** 2)
    return rss

# CLASS --------------------------
class Residuals():

    def __init__(self, population_size, random_seed):
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

        # Noyau de réproductibilité
        np.random.seed(self.random_seed)

        # Simulation des résidus selon une loi à priori
        if law == "normal":
            residuals_simulated = np.random.normal(loc=law_parameters[0], scale=law_parameters[1], size=self.population_size)
        elif law == "student":
            residuals_simulated = np.random.standard_t(df=law_parameters[0], size=self.population_size)
        elif law == "Pareto":
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

        # Synthèses des statistiques descriptives sous format dataframe
        statistics_values = [self._Count_, self._Mean_, self._Variance_, self._Q25_, self._Q50_, self._Q75_, self._Min_, self._Max_, self._Skewness_, self._Kurtosis_]
        statistics_names = ["Count", "Mean", "Variance", "Q25", "Q50", "Q75", "Min", "Max", "Skewness", "Kurtosis"]
        self.summary_residuals_statistics = pd.DataFrame({
            "Name": statistics_names,
            "Value": statistics_values
        })

        # sortie du résumé des statistiques descriptives
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

        # Histogramme des résidus simulés
        if figure == "histogram":
            sns.histplot(residuals_simulated, kde=True)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.title(f"Histogramme des residus simulés")
            plt.text(xmax * (1-0.05),
                     ymax * (1-0.05),
                     f"N={self._Count_}\n"
                     f" $\\mu$={self._Mean_}\n"
                     f" $\\sigma^{2}$={self._Variance_}\n"
                     f" $skewness$={self._Skewness_}\n"
                     f" $kurtosis$={self._Kurtosis_}",
                     ha='right',
                     va='top')
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            return plt.plot()

        # Boîte à moustache des résidus simulés
        elif figure == 'boxplot':
            sns.boxplot(residuals_simulated)
            plt.title(f"Boxplot des residus simulés")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.plot()

        # QQplot des résidus simulés
        elif figure == "qqplot":
            sm.qqplot(residuals_simulated, fit=True, line='45')
            plt.title("QQplot des residus simulées")
            plt.grid(True, alpha=0.25, linestyle='--')
            plt.show()
        else:
            raise ValueError(f"La valeurs associée à l'argument args: Figure est incorrecte")

class TargetPredictors():

    def __init__(self, population_size, true_coefficients):
        self.population_size = population_size
        self.true_coefficients = true_coefficients

    def Simulation(self, residuals_simulated):

        '''
        Objectif:
        ----------
        Fonction pour simuler les données d'entrées X et la variable y de la population.

        Arguments:
        ---------
        residuals_simulated : résidus simulés --> (list, np.ndarray).

        Sortie:
        -------
        data_simulated_population : données simulées de la population --> (pd.DataFrame).
        '''

        # variables associées au nombre de coefficients
        number_coefficient = len(self.true_coefficients)

        # Simulation des variables d'entrée X
        X = np.zeros(shape=(self.population_size, number_coefficient))
        for col in range(0,number_coefficient):
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

        # sortie des données simulée de la population
        return self.data_simulated_population


    def Sampling(self, population_simulated, sample_size):

        '''
        Objectif:
        ----------
        Fonction simuler les données de l'échantillon à partir des données de la population.

        Arguments:
        ---------
        population_simulated : données simulées de la population --> (pd.DataFrame).
        sample_size : taille souhaitée de l'échantillon simulé --> (int).

        Sortie:
        -------
        data_simulated_population : données simulées de la population --> (pd.DataFrame).
        '''

        data_simulated_sample = population_simulated.sample(sample_size)
        # sortie des données simulées de l'échantillon
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
        scatterplot : Nuage de points entre la variable cible y et les variables d'entrée X --> (plt).
        correlation_matrix : visualisation de le lineplot, scatterplot et correlation matrix des données simulées --> (plt).
        '''

        # Personnalisation de la palette des couleurs
        cmaps = plt.colormaps
        colors = cmaps.get_cmap("tab10").resampled(data_simulated.shape[1]).colors

        # Graphique en courbe
        if figure == "lineplot":
            fig, axs = plt.subplots()
            for i, col in enumerate(data_simulated.columns):
                axs.plot(data_simulated[col], label=col, linestyle="dotted", color=colors[i]) # personnalisation des couleurs
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
            # Matrice de corrélation
            data_simulated_corr = data_simulated.corr()
            sns.heatmap(data_simulated_corr, vmin=-1, vmax=1, cmap='RdBu_r', annot=True, cbar=True,
                        linewidths=1, annot_kws={'fontsize': 9})
            plt.title("Matrice de correlation")
            plt.show()
        else:
            raise ValueError("La valeurs associée à l'argument args: figure est incorrecte")

class Regularization():

    def __init__(self, predictors, target, alpha, intercept, random_seed):
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

        # Création du dataframe
        predictors_names = list(self.predictors.columns)

        # Intégration de l'intercept dans le vecteur des coefficients de régression
        if self.intercept == True:
            predictors_names.insert(index=0, object="Intercept")
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
                raise ValueError(f"La valeur associée à l'argument args: intercept est incorrecte")

            # Mise à jour du dataframe
            alpha_elem_name = f"{alpha_elem}"
            penalized_coefficients_table[alpha_elem_name] = result_complete

            # sortie du dataframe
        return penalized_coefficients_table


    def Visualization_Shrinking(self, penalized_coefficients_table, variables_selected, alpha_value_selected, figure):

        '''
        Objectif:
        ----------
        Fonction pour visualiser le rétrecissement des coefficients pénalisés en fonction de alpha.

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

        # Visualisation du rétrecissement selon une approche par courbe
        if figure == "curve":
            variables_selected == None
            alpha_value_selected == None

            # Manipulation de la base de données
            df = pd.DataFrame(penalized_coefficients_table)
            df.set_index(keys="Variables", drop=True, inplace=True)
            df_T = df.T

            # Sélection de la palette de couleurs
            cmaps = plt.colormaps
            colors = cmaps.get_cmap("tab10").resampled(df_T.shape[1]).colors

            # Visualisation
            fig, axs = plt.subplots()  # Automatisation de figsize
            for i, predictor in enumerate(df_T.columns):
                axs.plot(df_T.index, df_T[predictor], label=predictor, linestyle="-", alpha=1, linewidth=2, marker="*",
                         color=colors[i])  # Personalisation des couleurs
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

            # configuration de la figure
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
            raise TypeError("La valeur de l'argument <<purpose>> n'est pas correcte")

    # Calcul de l'écart-type, des statistiques, des pvaleurs
    # Evolution des coefficients
    # Evolution des écarts-types

# Fonction pour visualiser la relation entre les residus du modèle et les coefficients de régression
    def Vizualisation_Residuals(self, list_beta_1, list_beta_2, predictors, target, var_1, var_2, figure):

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

        # utilisation de la fonction rss_computing pour calculer la somme des carrés des résidus
        global rss_computing

        # configuration des variables employées
        X = predictors[[var_1, var_2]]
        X_values = X.values
        y_true = target
        b1 = list_beta_1
        b2 = list_beta_2
        rss = np.zeros(shape=(len(list_beta_1), len(list_beta_2)))

        # calculs de la somme des carrés des résidus
        j = 0
        for p in b1:
            i = 0
            for q in b2:
                rss[i, j] = rss_computing(beta=[p, q], X=X_values, y=y_true, intercept=5)
                i += 1
            j += 1
        B1, B2 = np.meshgrid(b1, b2)

        # courbe de niveaux
        if figure == "contour_map":
            plt.contour(B1, B2, rss, levels=50, linewidths=0.5, colors='yellow')
            cp = plt.contourf(B1, B2, rss, levels=100, cmap='magma')
            plt.colorbar(cp, label="RSS")
            plt.xlabel(f"$\\beta_{var_1[-1]}$")
            plt.ylabel(f"$\\beta_{var_2[-1]}$")
            plt.title("Carte de niveau du RSS")
            plt.grid(True, alpha=0.15)
            plt.show()

        # surface 3D
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


# Fonction pour visualiser la relation entre les coefficie

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
        cp = plt.contourf(Xi, Yi, Zi, levels=100, cmap='RdBu_r')  # RdBu_rcmap=''
        plt.colorbar(cp, label="Residual")
        # plt.plot(Coefs_B[0], Coefs_B[1], marker="o", color="darkred", label="True Coefficients")
        # plt.plot(coef_opt_ridge[var_1].values, coef_opt_ridge[var_2].values, marker="o", color="darkblue", label=f"Penalized Coefficients{model}")
        plt.xlabel(f"B_{var_1[-1]}")
        plt.ylabel(f"B_{var_2[-1]}")
        plt.title("Carte de niveau du RSS")
        plt.grid(True, alpha=0.15)
        # plt.legend()
        plt.show()