import numpy as np
import csv
import pandas as pd

class Adaline:
    def __init__(self,taxa_de_apredizagem=0.05, precisao=0.000001, max_epocas=1000, pesos=None):
        self.taxa_de_apredizagem = taxa_de_apredizagem
        self.precisao = precisao
        self.max_epocas = max_epocas
        self.pesos = pesos

    def EQM(self):
        EQM = 0

        for i in range(self.__X_treino.shape[0]):
            u = self.__pesos[0]

            for j in range(self.__X_treino.shape[1]):
                u += self.__pesos[j + 1] * self.__X_treino[i][j]

            y = self.sinal(u)

            EQM += (self.__y_treino[i] - y) ** 2 / self.__X_treino.shape[0]

        return EQM

    def sinal(self, u):
		return 1 if u >= 0 else 0


    def treinar(self, X_treino, y_treino):
        self.X_treino = X_treino
        self.y_treino = y_treino
        self.__pesos = np.random.rand(self.X_treino.shape[1] + 1)

        for i in range(self.X_treino.shape[0]):
            n_epocas = 0

            while n_epocas < self.__max_epocas:
                EQM_anterior = self.EQM()
                u = self.pesos[0]

                for j in range(self.__X_treino.shape[1]):
                    u += self.pesos[j + 1] * self.X_treino[i][j]

                y = 1 if u >= 0 else -1

                self.pesos[0] += self.taxa_de_apredizagem * (self.y_treino[i] - y)

                for j in range(self.X_treino.shape[1]):
                    self.pesos[j + 1] += self.taxa_de_apredizagem * (self.y_treino[i] - y) * self.X_treino[i][j]

                if abs(self.EQM() - EQM_anterior) <= self.precisao:
                    break

                n_epocas += 1

    def teste(self, X_teste, y_teste):
        self.X_teste = X_teste
        self.y_teste = y_teste
        n_erros = 0

        for i in range(self.X_teste.shape[0]):
            u = self.pesos[0]

            for j in range(self.X_teste.shape[1]):
                u += self.pesos[j + 1] * self.X_teste[i][j]

            y = 1 if u >= 0 else -1

            if self.y_teste[i] - y != 0:
                n_erros += 1

        self.percentual_de_acertos = 1 - n_erros / self.X_teste.shape[0]
