#
setwd("C:/Users/lajoi/Documents/1_PROGRAMME DE TRAVAIL/PROGRAMMATION/SimulationRidgeLasso")

# Installation des package
#install.packages('ridge')

# Importation des packages
library(ridge)
library(dplyr)
library(stargazer)
library(xtable)

# Aide
help(package="ridge")

# Chargement de la base de données
data <- read.csv('results_data/data_train.csv')
View(data)

# Régression Ridge
RegRidge <- linearRidge(y ~ ., data = data, lambda = 0.15, scaling = 'scale')
RidgeSummary <- summary(RegRidge)

# Stargazer
#stargazer(RidgeSummary)