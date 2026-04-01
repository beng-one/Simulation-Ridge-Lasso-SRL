#
setwd("C:/Users/lajoi/Documents/1_PROGRAMME DE TRAVAIL/PROGRAMMATION/SimulationRidgeLasso")

# Installation des package
#install.packages('ridge')

# Importation des packages
library(ridge)
library(dplyr)
library(stargazer)
library(xtable)
library(ggplot2)

# Aide
help(package="ridge")

# Chargement de la base de données
data <- read.csv('2_results_data/data_population.csv')
View(data)

# Régression Ridge
RegRidge <- linearRidge(y ~ ., data = data, lambda = 0.5, scaling = 'scale')
RidgeSummary <- summary(RegRidge)

# TrueCoefficients [0.69, 1.41, 2.73, -2.84, -3, 3.14, -7, 11]

N_brute <- 200                            # sample size
P_brute <- 2                              # number of predictors
V_brute <- 0.5 + 0.5 * diag(P_brute)            # covariance matrix of predictors
X_brute <- mvtnorm::rmvnorm(N_brute, sigma = V_brute) # generate predictors
y_brute <- X_brute %*% c(0.2, 0.5) + rnorm(N_brute)   # generate response variable


# plot
bgrid <- expand.grid(
  b1 = seq(-15, 15, length.out = 50),
  b2 = seq(-15, 15, length.out = 50)
)

opt <- coef(lm(y~., data = data))

y <- data$y
y <- data.matrix(y)
X <- data[,c('X_1', 'X_2')]
X <- data.matrix(X)

rss <- function(b, x, y, b0) {
  yhat <- b0 + X %*% b
  sum((y - yhat)^2)
}

rss <- mutate(bgrid, rss = apply(bgrid, 1, rss, x = X, y = y, b0 = 0)) # opt[1]

ggplot(rss, aes(b1, b2, z = rss)) +
  geom_contour_filled() +
  geom_point(aes(x = opt[2], y = opt[3]), color = "red", size = 3) +
  labs(
    title = "Residual sum of squares",
    subtitle = "Contours of residual sum of squares as a 
function of the regression coefficients",
    x = "Coefficient for predictor 1",
    y = "Coefficient for predictor 2"
  ) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "#fffbf2", colour = "transparent"))



# Install and load the plotly package
#install.packages("plotly")
library(plotly)

plot_ly(x = X, y = y, z = outer(b=bgrid,x= X, y=y, b0=0, rss), type = "surface")





## 3D SURFACE PLOT VERIFICATION

library(plotly)
library(dplyr)

y <- data$y
y <- data.matrix(y)
X <- data[,c('X_1', 'X_2')]
X <- data.matrix(X)

# 1. Création de la grille
bgrid <- expand.grid(
  b1 = seq(-4,4, length.out = 50),
  b2 = seq(-4,4, length.out = 50)
)

# 2. Définition de la fonction avec un nom unique (rss_func)
rss_func <- function(b, x_mat, y_vec, b0 = 0) {
  yhat <- b0 + x_mat %*% as.numeric(b)
  return(sum((y_vec - yhat)^2))
}

# 3. Application du calcul
# On utilise 'rss_func' (la fonction) pour créer la colonne 'rss_val' (la donnée)
bgrid <- bgrid %>%
  rowwise() %>%
  mutate(rss_val = rss_func(c(b1, b2), X, y))

# 4. Transformation en matrice pour plotly
z_matrix <- matrix(bgrid$rss_val, nrow = 50, ncol = 50)

# 5. Graphique
plot_ly(x = ~unique(bgrid$b1), y = ~unique(bgrid$b2), z = ~z_matrix) %>% 
  add_surface()









