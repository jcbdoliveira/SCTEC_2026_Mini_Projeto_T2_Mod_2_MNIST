import json
import os
import time

import joblib
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

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
    # 1. Divisão 80% Treino/Validação, 20% Teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 2. Normalização divide por 255 para ter 0.0 até 1.0
    x_treino_norm = x_treino.astype("float32") / 255.0    
    x_teste_norm = x_teste.astype("float32") / 255.0

    # 3. Formato 2D (N, 28, 28, 1) para a CNN
    x_treino_cnn = x_treino_norm.reshape(-1, 28, 28, 1)    
    x_teste_cnn = x_teste_norm.reshape(-1, 28, 28, 1)
    
    print(f"-> Treino: {x_treino_cnn.shape[0]} amostras | Formato CNN: {x_treino_cnn.shape}")    
    print(f"-> Teste:  {x_teste_cnn.shape[0]} amostras | Formato CNN: {x_teste_cnn.shape}")

    return x_treino_cnn, x_teste_cnn, y_treino, y_teste, x_treino_norm, x_teste_norm

def treinar_modelo_RandonForest(x_treino_norm, y_treino,):
    print("\n1. Treinando Modelo 1: Random Forest (n_estimators=100, max_depth=20)...")
    inicio_rf = time.time()
    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    modelo_rf.fit(x_treino_norm[:20000], y_treino[:20000])
    tempo_rf = time.time() - inicio_rf
    
    caminho_rf =  os.path.join("models/", "random_forest_mnist.joblib")
    joblib.dump(modelo_rf, caminho_rf)
    print(f"   -> Concluído em {tempo_rf:.2f}s | Salvo em '{caminho_rf}'")
    return tempo_rf, caminho_rf

def treinar_modelo_SVM(x_treino_norm, y_treino,):
    print("\n2. Treinando Modelo 2: Support Vector Machine (C=5.0, kernel='rbf')...")
    inicio_svm = time.time()
    modelo_svm = SVC(
        C=5.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    modelo_svm.fit(x_treino_norm[:15000], y_treino[:15000])
    tempo_svm = time.time() - inicio_svm
    caminho_svm = os.path.join("models/", "svm_mnist.joblib")
    joblib.dump(modelo_svm, caminho_svm)
    print(f"   -> Concluído em {tempo_svm:.2f}s | Salvo em '{caminho_svm}'")
    return tempo_svm, caminho_svm

def criar_modelo_CNN():
    modelo = models.Sequential([
        # Bloco 1: Convolução 2D (32 filtros 3x3) + ReLU + MaxPool (2x2)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name="conv2d_camada1"),
        layers.MaxPooling2D((2, 2), name="maxpool_camada1"),
        
        # Bloco 2: Convolução 2D (64 filtros 3x3) + ReLU + MaxPool (2x2)
        layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_camada2"),
        layers.MaxPooling2D((2, 2), name="maxpool_camada2"),
        
        # Bloco 3: Achatar (Flatten) para vetor denso
        layers.Flatten(name="flatten_tensor"),
        
        # Bloco 4: Camada Densa com 128 neurônios + Dropout de 25% para evitar overfitting
        layers.Dense(128, activation='relu', name="densa_128"),
        layers.Dropout(0.25, name="dropout_regularizacao"),
        
        # Camada de Saída: 10 classes com ativação Softmax
        layers.Dense(10, activation='softmax', name="saida_softmax")
    ], name="CNN_MNIST_WhiteBox")
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return modelo

def treinar_modelo_CNN(x_treino_cnn, y_treino):
    modelo_cnn = criar_modelo_CNN()
    modelo_cnn.summary()

    inicio_cnn = time.time()
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join("models/", "cnn_mnist.keras"),
        save_best_only=True,
        monitor='val_loss',
        mode='max'
    )

    historico_treino = modelo_cnn.fit(
        x_treino_cnn, y_treino,
        epochs=10,
        batch_size=64,
        validation_split=0.15,
        callbacks=[checkpoint_cb],
        verbose=1
    )
    tempo_cnn = time.time() - inicio_cnn    
    caminho_cnn = os.path.join("models/", "cnn_mnist.keras")
    print(f"   -> Concluído em {tempo_cnn:.2f}s | Salvo em '{caminho_cnn}'")
    return tempo_cnn, caminho_cnn

def avaliar_modelos(x_teste_norm, y_teste, x_teste_cnn, tempos_treino):
    print("\n3. Avaliando Modelos treinados...")
    # Carregar modelos salvos
    modelo_rf = joblib.load(os.path.join("models/", "random_forest_mnist.joblib"))
    modelo_svm = joblib.load(os.path.join("models/", "svm_mnist.joblib"))
    modelo_cnn = tf.keras.models.load_model(os.path.join("models/", "cnn_mnist.keras"))

    pred_rf = modelo_rf.predict(x_teste_norm)
    pred_svm = modelo_svm.predict(x_teste_norm)
    prob_cnn = modelo_cnn.predict(x_teste_cnn, batch_size=128)
    pred_cnn = np.argmax(prob_cnn, axis=1)

    todos_pred = {
        'Random Forest': pred_rf,
        'SVM(RBF)': pred_svm,
        'CNN (Deep Learning)': pred_cnn
    }

    tabela_resultados = []
    plt.figure(figsize=(18, 5))

    for idx, (nome, preds) in enumerate(todos_pred.items(), 1):
        acc = accuracy_score(y_teste, preds)
        prec = precision_score(y_teste, preds, average='weighted')
        rec = recall_score(y_teste, preds, average='weighted')
        f1 = f1_score(y_teste, preds, average='weighted')
        
        tabela_resultados.append({
            'Modelo': nome,
            'Acurácia Global': f"{acc * 100:.2f}%",
            'Precisão Ponderada': f"{prec * 100:.2f}%",
            'Recall (Sensibilidade)': f"{rec * 100:.2f}%",
            'F1-Score': f"{f1 * 100:.2f}%",
            'Tempo Treino (s)': f"{tempos_treino[nome]:.2f}s",
            'Erros no Teste': int(np.sum(y_teste != preds))
        })
        
        # Matriz de Confusão 10x10
        matriz_conf = confusion_matrix(y_teste, preds)
        plt.subplot(1, 3, idx)
        sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='YlOrBr' if idx==3 else 'Blues', cbar=False,
                    xticklabels=range(10), yticklabels=range(10))
        plt.title(f"Matriz 10x10: {nome}\nAcurácia: {acc*100:.2f}%", fontweight='bold')
        plt.xlabel("Dígito Predito")
        plt.ylabel("Dígito Real")

    plt.tight_layout()
    plt.savefig("images/fase4_matrizes_confusao_comparativas.png", dpi=300)
    plt.close()
    print("Matrizes de Confusão salvas: 'fase4_matrizes_confusao_comparativas.png'")

    df_metricas = pd.DataFrame(tabela_resultados)
    print("\n=== TABELA COMPARATIVA CONSOLIDADA DE DESEMPENHO ===")
    print(df_metricas.to_string(index=False))

    # Salvar metadados em json na pasta models
    with open(os.path.join("models/", "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data_treinamento": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tamanho_teste": len(y_teste),
            "resultados": tabela_resultados,
            "classes": list(range(10))
        }, f, indent=2, ensure_ascii=False)

    print(f"Metadados salvos em '{os.path.join('models/', 'metadata.json')}'")