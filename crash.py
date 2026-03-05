# Fonction pour simuler les coefficiens de régression estimés selon une loi uniforme en fonction des variables d'entrée sélectionnées.
def Estimated_Coefficients_Framed(Estimated_Coefficients_init, Variability, Number_Estimated_Coefficient, framed):

    '''
     Objectif:
    ---------
    Fonction pour simuler les coefficiens de régression estimés selon une loi uniforme en fonction des variables d'entrée sélectionnées.

    Arguments:
    ----------
    Estimated_Coefficients_init :
    Variability :
    Number_Estimated_Coefficient :
    framed :
    '''

    #if not isinstance(Estimated_Coefficients_init, np.ndarray):
    #    raise TypeError("Le type de 'Estimated_Coefficients_init' n'est pas le bon. Il doit être de type 'np.ndarray'.")
    #if not isinstance(Obs_True, np.ndarray):
    #    raise TypeError("Le type de 'Obs_True' n'est pas le bon. Il doit être de type 'np.ndarray'.")
    #if not isinstance(Inputs, pd.DataFrame):
    #    raise TypeError("Le type de 'Inputs' n'est pas le bon. Il doit être de type 'pd.DataFrame'.")

    DataFrame_Estimated_Coefficient = pd.DataFrame()

    # Commentaire : Il se peut que certains coéfficients de la régression lasso soient nuls, par conséquent, il est préférable de mettre un garde-fou.
    for enum, i in enumerate(Estimated_Coefficients_init):
        if i != 0 :
            estimated_coef = Estimated_Coefficients_init[enum]

            # borne inférieure
            # --- born_inf = estimated_coef*(1-Variability)
            # --- interval_inf = np.linspace(start=born_inf,stop=estimated_coef, num=Number_Estimated_Coefficient, endpoint=False)
            # --- interval_inf = np.random.uniform(low=born_inf, high=estimated_coef, size=Number_Estimated_Coefficient)
            interval_inf = np.random.uniform(low=-framed, high=framed, size=Number_Estimated_Coefficient)

            # borne supérieure
            # --- born_sup = estimated_coef*(1+Variability)
            # --- interval_sup = np.linspace(start=estimated_coef, stop=born_sup, num=Number_Estimated_Coefficient, endpoint=True)
            # --- interval_sup = np.random.uniform(low=estimated_coef, high=born_sup, size=Number_Estimated_Coefficient)
            # --- interval_inf = np.random.uniform(low=-framed, high=framed, size=Number_Estimated_Coefficient)

            # Interval complet
            # --- noise = np.random.normal(0, abs(estimated_coef) * Variability / 5, Number_Estimated_Coefficient * 2)
            # --- interval_complete = np.concat((interval_inf, interval_sup)) + noise
            # --- interval_complete = np.concat((interval_inf, interval_sup))
            interval_complete = interval_inf
            DataFrame_Estimated_Coefficient[f"X_{enum+1}"] = interval_complete
            print(f"\n Fin de l'encadrement du coefficient estimé de la variable X_{enum+1}\n -----------")

    return DataFrame_Estimated_Coefficient
