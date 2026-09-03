from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
import random
import pandas as pd
from sklearn.model_selection import train_test_split

#FASE 01: EDA 
#Carregamento do dataset
def carregar_dataset():    
    console = Console()
    
    # Cria uma animação de carregamento para indicar que esta fazendo download
    with console.status("[bold green]Baixando dataset MNIST... (Isso pode levar alguns minutos)", spinner="dots"):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32)  
        y = mnist.target.astype(int)      
        
    console.print("[bold blue]✓ Dataset carregado com sucesso![/bold blue]")
  
    print(f"-> Dimensionalidade de X (pixels): {X.shape}")
    print(f"   Significado: {X.shape[0]} imagens de 28x28 (784 pixels).")    
    return X, y    

def visualizar_balanceamento_classes(X, y):
    contagem_classes = pd.Series(y).value_counts().sort_index()
    print("\nDistribuição de amostras por dígito (0 a 9):")
    print(contagem_classes)

    plt.figure(figsize=(9, 4))
    plt.bar(contagem_classes.index, contagem_classes.values, color='#5A5A40', edgecolor='#33332D')
    plt.title("Balanceamento das classes", fontsize=13, fontweight='bold')
    plt.xlabel("Dígito", fontsize=11)
    plt.ylabel("Qtde imagens", fontsize=11)
    plt.xticks(range(10))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("images/balanceamento_classes.png", dpi=300)
    plt.close()
    print("Gráfico salvo em : 'images/balanceamento_classes.png'")

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
    plt.savefig(f"images/grade_exemplos{indice}.png", dpi=300)
    plt.gcf().canvas.manager.set_window_title(f"Amostra aleatória: amostra {indice} de {X.shape[1]}")
    plt.close()
    print(f"Gráfico salvo em : 'images/grade_exemplos{indice}.png")

def processamento_e_separacao(X, y):
    # 1. Divisão Estratificada (80% Treino/Validação, 20% Teste)
    x_treino_val, x_teste, y_treino_val, y_teste = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 2. Normalização correta usando as variáveis de imagem (X) e tipo float32
    x_treino_norm = x_treino_val.astype("float32") / 255.0    
    x_teste_norm = x_teste.astype("float32") / 255.0

    # 3. Formato 2D (N, 28, 28, 1) para a CNN
    x_treino_cnn = x_treino_norm.reshape(-1, 28, 28, 1)    
    x_teste_cnn = x_teste_norm.reshape(-1, 28, 28, 1)

    # 4. Prints informativos ajustados
    print(f"-> Treino: {x_treino_cnn.shape[0]} amostras | Formato CNN: {x_treino_cnn.shape}")    
    print(f"-> Teste:  {x_teste_cnn.shape[0]} amostras | Formato CNN: {x_teste_cnn.shape}")

    # 5. Retorno dos dados para uso posterior no modelo
    return x_treino_cnn, x_teste_cnn, y_treino_val, y_teste
