# IMPORTATION DES LIBRAIRIES --------------------------

# Manipulation des données
import math
import numpy as np
import pandas as pd
import scipy

# Visualitation
import matplotlib.pyplot as plt
import seaborn as sns

# Modèles de régression
import sklearn
from hyperframe.frame import DataFrame
from numpy.random.mtrand import RandomState
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Vérification des résultats de sklearn
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import cvxopt

# Répertoire de travail et gestion d'erreur
import os

# Gestion des erreurs et typage statistique
import warnings
import typing
import mypy

# CLASS --------------------------
class Residuals():

    def __init__(self, population_size, random_seed):
        self.population_size = population_size
        self.random_seed = random_seed

    def Simulation(self, law="normal", law_parameters=[0,1]):

        np.random.seed(self.random_seed)
        if law == "normal":
            residuals_simulated = np.random.normal(loc=law_parameters[0], scale=law_parameters[1], size=self.population_size)
        elif law == "student":
            residuals_simulated = np.random.standard_t(df=law_parameters[0], size=self.population_size)
        elif law == "Pareto":
            residuals_simulated = np.random.pareto(a=law_parameters[0], size=self.population_size) # forme de la distribution (a>=0)
        elif law == "weibull":
            residuals_simulated = np.random.weibull(a=law_parameters[0], size=self.population_size)
        return residuals_simulated

    def Statistics(self, residuals_simulated, digits):


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
        return self.summary_residuals_statistics

    def Visualization(self, residuals_simulated, figure):

        # Histogramme
        if figure == "histogram":
            sns.histplot(residuals_simulated, kde=True)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.title(f"Histogramme des residus simulés")
            plt.text(xmax * (1-0.05),
                     ymax * (1-0.05),
                     f"N={self._Count_}\n"
                     f" $\\mu$={self._Mean_}\n"
                     f" $\\sigma^{2}$={self._Variance_}\n"
                     f" skewness={self._Skewness_}\n"
                     f" kurtosis={self._Kurtosis_}",
                     ha='right',
                     va='top')
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            return plt.plot()

        # Boîte à moustache
        elif figure == 'boxplot':
            sns.boxplot(residuals_simulated)
            plt.title(f"Boxplot des residus simulés")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.plot()

        # qqplot
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
        return self.data_simulated_population



    def Sampling(self, population_simulated, sample_size):
        data_simulated_sample = population_simulated.sample(sample_size)
        return data_simulated_sample

    def Visualization(self, data_simulated, figure):

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

    def Penalized_Regression(self, model):

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

        # Modèle de régression Ridge en fonction de alpha
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
        return penalized_coefficients_table


    def Visualization_Shrinking(self, penalized_coefficients_table):

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

    def Predict(self, penalized_coefficients_table, variables_selected, alpha_value, true_target, predictors, purpose):

        if 'Intercept' in penalized_coefficients_table["Variables"].tolist():
            intercept_filter = penalized_coefficients_table["Variables"].isin(["Intercept"])
            intercept_value = penalized_coefficients_table.loc[intercept_filter, alpha_value][0]
            penalized_coefficients_shape = penalized_coefficients_table.shape[0] - 1

        else:
            intercept_value = 0
            penalized_coefficients_shape = penalized_coefficients_table.shape[0]
        variables_selected_filter = penalized_coefficients_table["Variables"].isin(variables_selected)

        penalized_coefficients_filter = penalized_coefficients_table.loc[variables_selected_filter, alpha_value]
        penalized_coefficients_values = penalized_coefficients_filter.tolist()
        penalized_coefficients_index = penalized_coefficients_filter.index.tolist()
        penalized_coefficients_init = np.zeros(penalized_coefficients_shape)

        for i in range(len(penalized_coefficients_index)):
            penalized_coefficients_init[penalized_coefficients_index[i]] = penalized_coefficients_values[i]
        inputs_array = predictors.values

        penalized_coefficients_init_reshape = np.array(penalized_coefficients_init).reshape(1,penalized_coefficients_shape)
        predicted_target = inputs_array @ penalized_coefficients_init_reshape.T + intercept_value * np.ones(
            shape=(predictors.shape[0], 1))

        if purpose == "predicting":
            return predicted_target
        elif purpose == "residuals_of_square":
            residuals = np.sum((true_target - predicted_target) ** 2)
            return residuals
        elif purpose == "residuals":
            residuals = (true_target - predicted_target)**2
            return residuals
        else:
            raise TypeError("La valeur de l'argument <<purpose>> n'est pas correcte")

    # Calcul de l'écart-type, des statistiques, des pvaleurs
    # Evolution des coefficients
    # Evolution des écarts-types


    def gradient_residual_plot(self, data_simulated_coefficients, var_1, var_2, residual, model):

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

    # Fonction pour simuler les coefficiens de régression estimés selon une loi uniforme en fonction des variables d'entrée sélectionnées.
    def Simulation_Penalized_Coefficients(penalized_coefficients_table, variables_selected, alpha_value,
                                          true_coefficients, number_simulated_penalized_coefficients, model):

        data_simulated_coefficients = pd.DataFrame()

        variables_selected_filter = penalized_coefficients_table["Variables"].isin(variables_selected)
        penalized_coefficients_filter = penalized_coefficients_table.loc[variables_selected_filter, alpha_value]

        # Commentaire : Il se peut que certains coéfficients de la régression lasso soient nuls, par conséquent, il est préférable de mettre un garde-fou.
        for enum, i in enumerate(penalized_coefficients_filter):
            if i != 0:
                if model == "ridge":
                    interval_complete = np.random.normal(true_coefficients[enum], alpha_value,
                                                         number_simulated_penalized_coefficients)
                elif model == "lasso":
                    interval_complete = np.random.laplace(true_coefficients[enum], alpha_value,
                                                          number_simulated_penalized_coefficients)
                else:
                    raise ValueError("La valeurs associée à l'argument 'Model' est incorrect.")
                data_simulated_coefficients[f"X_{enum + 1}"] = interval_complete
                print(f"\n Fin de l'encadrement du coefficient estimé de la variable X_{enum + 1}\n -----------")

        for line_index in data_simulated_coefficients.index:
            beta_penalized_simulated = data_simulated_coefficients.iloc[line_index, :]
            beta_penalized_simulated_array = np.array(beta_penalized_simulated.values)
            y_simulated = input_array @ beta_penalized_simulated_array
            residuals_simulated = np.sum((output - y_simulated) ** 2)
            list_residual_sum_square.append(residuals_simulated)
            print(residual)
            print(f"Somme des carrés des residus du Cycle : {line} \n-----------")
        data_simulated_coefficients['RSS'] = list_residual_sum_square
        return data_simulated_coefficients



