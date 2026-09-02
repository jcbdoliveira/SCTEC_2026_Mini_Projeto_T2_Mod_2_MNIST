#FASE 01: EDA 
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
import random

#Carregamento do dataset
def carregar_dataset():    
    console = Console()
    
    # Cria uma animação de carregamento para indicar que esta fazendo download
    with console.status("[bold green]Baixando e processando o MNIST... (Isso pode levar alguns minutos)", spinner="dots"):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32)  # Matriz de pixels (70000, 784)
        y = mnist.target.astype(int)       # Rótulos (70000,)
        
    console.print("[bold blue]✓ Dataset carregado com sucesso![/bold blue]")
  
    print(f"Shape X: {X.shape} | Shape y: {y.shape}")
    visualizar_dataset_carregado(X, y)  

#Grade Visual 2x5 com Matplotlib
def visualizar_dataset_carregado(X, y):
    indice = random.randint(1, X.shape[1])
    plt.figure(figsize=(10, 4))
    for digito in range(10):        
        idx = np.where(y == digito)[0][indice]
        plt.subplot(2, 5, digito + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray_r')
        plt.title(f"Digito: {digito}")
        plt.axis('off')
    plt.tight_layout()    
    plt.gcf().canvas.manager.set_window_title(f"Amostra aleatória: amostra {indice} de {X.shape[1]}")
    plt.show()