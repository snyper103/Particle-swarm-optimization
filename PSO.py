# Gabriel Alexandre de Souza Braga
# Versão do python utilizada: 3.6.5 64-bit ('base': conda)
# Trabalho de particle swarm optimization (PSO) da matéria de IA da UTFPR

import random                       # para criar números aleatórios
import math                         # para usar a função matemáticas para a função objetivo
import matplotlib.pyplot as plt     # para plotar os gráficos
import numpy as np                  # para função objetivo

# função objetivo e parâmetros de personalização
def funcaoObjetivo (x):
    z = -4*math.fabs(math.exp(math.fabs(math.cos((1/200) * x[0]**2 + 1/200 * x[1]**2))) * math.sin(x[0]) * math.cos(x[1]))    # função que achei no gloogle e achei difícil
    #z = (x[0]**2 - 10 * np.cos(2 * math.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * math.pi * x[1])) + 20  # função de rastrigin

    return z

limites = [(-10,10),(-10,10)]   # limite superior e inferior da função (para o problema de rastrigin use [(-4,4),(-4,4)])
limSize = 2                 # tamanho do limite
flag = 0                    # flag para determinar se vai encontrar o maior valor ou menos, para maior flag = 1, menor flag = 0
numParticulas = 30          # numero de particulas
iMax = 1000                 # numero maximo de interações
w = 1.05                    # constante de peso ou inercia
c1 = 2                      # constante cognitiva
c2 = 2                      # constante social

# ----------- Criando o objeto partícula -----------
class Particula:
    def __init__(self, limites):  # Construtor do nosso objeto
        self.pos = []                               # posição da partícula
        self.velocidade = []                        # velocidade da partícula
        self.melhorPosLocal = []                    # melhor posição da partícula local
        self.inicialMelhorPosLocal = valorInicial   # valor inicial da melhor posição da partícula local da função objetivo
        self.inicialPos = valorInicial              # valor inicial da posição da partícula da função objetivo

        for i in range (limSize):
            self.pos.append(random.uniform(limites[i][0], limites[i][1]))   # gera números randomicos para a posição inicial
            self.velocidade.append(random.uniform(-1,1))                    # gera números randomicos para a velocidade

    def validaMelhorPosicao(self, funcaoObjetivo):
        self.inicialPos = funcaoObjetivo(self.pos)

        if flag == 0:
            if self.inicialPos < self.inicialMelhorPosLocal:
                self.melhorPosLocal = self.pos                  # atualiza a melhor posição local
                self.inicialMelhorPosLocal = self.inicialPos    # atualiza a melhor posição local inicial

        if flag == 1:
            if self.inicialPos > self.inicialMelhorPosLocal:
                self.melhorPosLocal = self.pos                  # atualiza a melhor posição local
                self.inicialMelhorPosLocal = self.inicialPos    # atualiza a melhor posição local inicial

    def atualizaVelocidade(self, melhorPosGlobal):
        rand = [0.0, 0.0]

        for i in range (limSize):
            rand[0] = random.random() 
            rand[1] = random.random()

            velocidadeCognitiva = c1 * rand[0] * (self.melhorPosLocal[i] - self.pos[i])
            velocidadeSocial = c2 * rand[1] * (melhorPosGlobal[i] - self.pos[i])
            self.velocidade[i] = w * self.velocidade[i] + velocidadeCognitiva + velocidadeSocial
    
    def atualizaPosicao(self, limites):
        for i in range (limSize):
            self.pos[i] += self.velocidade[i]

            # verifica se o limite inferior não foi ultrapassado, caso tenhas sido ele satura a posição
            if self.pos[i] < limites[i][0]:
                self.pos[i] = limites[i][0]

            # verifica se o limite superio não foi ultrapassado, caso tenhas sido ele satura a posição
            if self.pos[i] > limites[i][1]:
                self.pos[i] = limites[i][1]

# ----------- Criando o objeto PSO -----------
class PSO():
    def __init__(self, funcaoObjetivo, limites, numParticulas, iMax):    # Construtor do nosso objeto
        inicialMelhorPosGlobal = valorInicial   # valor inicial da melhor posição da partícula global da função objetivo
        melhorPosGlobal = []                    # melhor posição da partícula global
        enxameDeParticulas = []                 # nossa lista estática do objeto partícula
        aux = []                                # variavel auxiliar

        # inicializa o vetor enxameDeParticulas
        for i in range (numParticulas):
            enxameDeParticulas.append(Particula(limites))

        # efetua as interações até concluir todas elas e encontra a melhor solução global e local
        for i in range (iMax):
            for j in range (numParticulas):
                enxameDeParticulas[j].validaMelhorPosicao(funcaoObjetivo)

                if flag == 0:
                    if enxameDeParticulas[j].inicialPos < inicialMelhorPosGlobal:
                        melhorPosGlobal = list(enxameDeParticulas[j].pos)                    # atualiza a melhor posição global
                        inicialMelhorPosGlobal = float(enxameDeParticulas[j].inicialPos)     # atualiza a melhor posição global inicial

                if flag == 1:
                    if enxameDeParticulas[j].inicialPos > inicialMelhorPosGlobal:
                        melhorPosGlobal = list(enxameDeParticulas[j].pos)                    # atualiza a melhor posição global
                        inicialMelhorPosGlobal = float(enxameDeParticulas[j].inicialPos)     # atualiza a melhor posição global inicial

            for j in range (numParticulas):
                enxameDeParticulas[j].atualizaVelocidade(melhorPosGlobal)   # atualiza velocidade das particulas
                enxameDeParticulas[j].atualizaPosicao(limites)              # atualiza posição das particulas

            aux.append(inicialMelhorPosGlobal)  # salva o valor da melhor posição global inicial

        # imprime os resultados e plota o gráfico
        print('Posição ótimo global: ', melhorPosGlobal)
        print('Solução ótimo global: ', inicialMelhorPosGlobal)
        print('Gráfico de convergência de soluções: ')
        plt.subplots()
        plt.plot(aux)
        plt.xlabel('iterações')
        plt.ylabel('Melhores soluções')
        plt.title('Gráfico de convergência de soluções')
        plt.grid()
        plt.show()

# ----------- Definindo valor inicial -----------
if flag == 0:
    valorInicial = float("inf") # para o problema de minimização (encontrar o menor valor da função)

if flag == 1:
    valorInicial = -float("inf") # para o problema de maximização (encontrar o maior valor da função)

# ----------- Criando PSO -----------
PSO(funcaoObjetivo, limites, numParticulas, iMax)
