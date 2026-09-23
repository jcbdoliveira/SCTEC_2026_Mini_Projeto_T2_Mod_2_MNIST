import json
import os
import time
import cv2
import joblib
from sklearn.datasets import fetch_openml
import matplotlib
matplotlib.use('Agg') 
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# FASE 01: EDA & CARREGAMENTO
# ==========================================
def carregar_dataset():    
    console = Console()
    with console.status("[bold green]Baixando dataset MNIST... (Isso pode levar alguns minutos)", spinner="dots"):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32)  
        y = mnist.target.astype(int)      
        
    console.print("[bold blue]✓ Dataset carregado com sucesso![/bold blue]")
    print(f"-> Dimensionalidade de X (pixels): {X.shape}")
    print(f"   Significado: {X.shape[0]} imagens de 28x28 (784 pixels).")    
    return X, y    

def visualizar_balanceamento_classes(X, y, modelo):
    os.makedirs(f"images/{modelo}", exist_ok=True)
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
    plt.savefig(f"images/{modelo}/balanceamento_classes.png", dpi=300)
    plt.close()
    print(f"Gráfico salvos em: 'images/{modelo}/balanceamento_classes.png'")

def visualizar_dataset_carregado(X, y, modelo):
    os.makedirs(f"images/{modelo}", exist_ok=True)
    indice = random.randint(1, X.shape[1])
    plt.figure(figsize=(10, 4))
    for digito in range(10):        
        idx = np.where(y == digito)[0][indice]
        plt.subplot(2, 5, digito + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray_r')
        plt.title(f"Digito: {digito}")
        plt.axis('off')
    plt.tight_layout()    
    plt.savefig(f"images/{modelo}/grade_exemplos{indice}.png", dpi=300)
    plt.close()
    print(f"Gráfico salvo em: 'images/{modelo}/grade_exemplos{indice}.png'")

# ==========================================
# FASE 02: PRÉ-PROCESSAMENTO & SEPARAÇÃO
# ==========================================
def processamento_e_separacao(X, y, modelo='V1'):
    os.makedirs(f"images/{modelo}", exist_ok=True)
    os.makedirs(f"models/{modelo}", exist_ok=True)
    
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

    if modelo == 'V2':
        print("-> Gerando imagens com traços finos (Erosão)...")
        x_treino_extra = []
        kernel = np.ones((2, 2), np.uint8)
        for img in x_treino_cnn:
            img_eroded = cv2.erode(img[:, :, 0], kernel, iterations=1)
            x_treino_extra.append(np.expand_dims(img_eroded, -1))
        x_treino_extra = np.array(x_treino_extra)
        
        numero_aleatorio = random.randint(0, len(x_treino_cnn) - 1)
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(x_treino_cnn[numero_aleatorio, :, :, 0], cmap='gray')
        plt.title("Original (MNIST)")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(x_treino_extra[numero_aleatorio, :, :, 0], cmap='gray')
        plt.title("V2 (Erosão / Fino)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"images/{modelo}/{numero_aleatorio}_comparativo_processamento.png", dpi=150)
        plt.close()

        x_treino_expandido = np.concatenate((x_treino_cnn, x_treino_extra), axis=0)
        y_treino_expandido = np.concatenate((y_treino, y_treino), axis=0)
        x_treino_norm_expandido = x_treino_expandido.reshape(-1, 784)
        return x_treino_expandido, x_teste_cnn, y_treino_expandido, y_teste, x_treino_norm_expandido, x_teste_norm
        
    elif modelo == 'V3':
        print("-> [V3] Gerando imagens com traços robustos (Dilação) e suavizados (Blur)...")
        x_treino_extra = []
        kernel = np.ones((2, 2), np.uint8)
        for img in x_treino_cnn:
            img_255 = (img[:, :, 0] * 255).astype(np.uint8)
            img_dilated = cv2.dilate(img_255, kernel, iterations=1)
            img_blurred = cv2.GaussianBlur(img_dilated, (3, 3), 0)
            img_final = img_blurred.astype("float32") / 255.0
            x_treino_extra.append(np.expand_dims(img_final, -1))
        x_treino_extra = np.array(x_treino_extra)
        
        numero_aleatorio = random.randint(0, len(x_treino_cnn) - 1)
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(x_treino_cnn[numero_aleatorio, :, :, 0], cmap='gray')
        plt.title("Original (MNIST)")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(x_treino_extra[numero_aleatorio, :, :, 0], cmap='gray')
        plt.title("V3 (Dilação + Blur)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"images/{modelo}/{numero_aleatorio}_comparativo_processamento.png", dpi=150)
        plt.close()

        x_treino_expandido = np.concatenate((x_treino_cnn, x_treino_extra), axis=0)
        y_treino_expandido = np.concatenate((y_treino, y_treino), axis=0)
        x_treino_norm_expandido = x_treino_expandido.reshape(-1, 784)
        return x_treino_expandido, x_teste_cnn, y_treino_expandido, y_teste, x_treino_norm_expandido, x_teste_norm

    print(f"-> Treino Original: {x_treino_cnn.shape[0]} amostras")   
    print(f"-> Teste (Mantido original): {x_teste_cnn.shape[0]} amostras | Formato CNN: {x_teste_cnn.shape}")
    return x_treino_cnn, x_teste_cnn, y_treino, y_teste, x_treino_norm, x_teste_norm

# ==========================================
# FASE 03: TREINAMENTO DOS MODELOS
# ==========================================
def treinar_modelo_RandonForest(x_treino_norm, y_treino, modelo):
    print("\n1. Treinando Modelo 1: Random Forest (n_estimators=100, max_depth=20)...")
    inicio_rf = time.time()
    modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    modelo_rf.fit(x_treino_norm[:20000], y_treino[:20000])
    tempo_rf = time.time() - inicio_rf
    caminho_rf = os.path.join("models", modelo, "random_forest_mnist.joblib")
    joblib.dump(modelo_rf, caminho_rf)
    print(f"   -> Concluído em {tempo_rf:.2f}s | Salvo em '{caminho_rf}'")
    return tempo_rf, caminho_rf

def treinar_modelo_SVM(x_treino_norm, y_treino, modelo):
    print("\n2. Treinando Modelo 2: Support Vector Machine (C=5.0, kernel='rbf')...")
    inicio_svm = time.time()
    modelo_svm = SVC(C=5.0, kernel='rbf', gamma='scale', random_state=42)
    modelo_svm.fit(x_treino_norm[:15000], y_treino[:15000])
    tempo_svm = time.time() - inicio_svm
    caminho_svm = os.path.join("models", modelo, "svm_mnist.joblib")
    joblib.dump(modelo_svm, caminho_svm)
    print(f"   -> Concluído em {tempo_svm:.2f}s | Salvo em '{caminho_svm}'")
    return tempo_svm, caminho_svm

# --- Arquitetura Original (V1) ---
def criar_modelo_CNN_V1():
    modelo = models.Sequential([
        layers.Input(shape=(28, 28, 1), name="entrada_imagem"),
        layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_camada1"),
        layers.MaxPooling2D((2, 2), name="maxpool_camada1"),
        layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_camada2"),
        layers.MaxPooling2D((2, 2), name="maxpool_camada2"),
        layers.Flatten(name="flatten_tensor"),
        layers.Dense(128, activation='relu', name="densa_128"),
        layers.Dropout(0.25, name="dropout_regularizacao"),
        layers.Dense(10, activation='softmax', name="saida_softmax")
    ], name="CNN_MNIST_V1")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

# --- Nova Arquitetura Otimizada (V4 - Alta Acurácia) ---
def criar_modelo_CNN_V4():
    modelo = models.Sequential([
        layers.Input(shape=(28, 28, 1), name="entrada_imagem"),
        
        # Bloco Convolucional 1 (Duplo)
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloco Convolucional 2 (Duplo)
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classificador
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name="CNN_MNIST_Otimizada_V4")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return modelo

def treinar_modelo_CNN(x_treino_cnn, y_treino, modelo):
    print(f"\n3. Treinando Modelo 3: Convolutional Neural Network ({modelo})...")
    caminho_cnn = os.path.join("models", modelo, "cnn_mnist.keras")
    
    if modelo == 'V4':
        modelo_cnn = criar_modelo_CNN_V4()
        modelo_cnn.summary()
        
        # Configurando Data Augmentation em tempo real para V4
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        datagen.fit(x_treino_cnn)
        
        callbacks_lista = [
            callbacks.ModelCheckpoint(
                filepath=caminho_cnn,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        inicio_cnn = time.time()
        
        # Treinamento com gerador aumentado e divisão manual de validação (15% para validação)
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_treino_cnn, y_treino,
            test_size=0.15,
            random_state=42,
            stratify=y_treino
        )
        
        historico_treino = modelo_cnn.fit(
            datagen.flow(x_tr, y_tr, batch_size=64),
            epochs=20,  # O EarlyStopping gerencia o encerramento seguro
            validation_data=(x_val, y_val),
            callbacks=callbacks_lista,
            verbose=1
        )
    else:
        # Pipeline Original (V1, V2, V3)
        modelo_cnn = criar_modelo_CNN_V1()
        modelo_cnn.summary()
        
        checkpoint_cb = callbacks.ModelCheckpoint(
            filepath=caminho_cnn,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        
        inicio_cnn = time.time()
        
        historico_treino = modelo_cnn.fit(
            x_treino_cnn, y_treino,
            epochs=20,
            batch_size=64,
            validation_split=0.15,
            callbacks=[checkpoint_cb],
            verbose=1
        )
    
    tempo_cnn = time.time() - inicio_cnn
    print(f" -> Concluído em {tempo_cnn:.2f}s | Salvo em '{caminho_cnn}'")
    return tempo_cnn, caminho_cnn

# ==========================================
# FASE 04: AVALIAÇÃO & MÉTRICAS
# ==========================================

def avaliar_modelos(x_teste_norm, y_teste, x_teste_cnn, tempos_treino, modelo):
    print("\n4. Avaliando Modelos treinados...")
    
    modelo_rf = joblib.load(os.path.join("models", modelo, "random_forest_mnist.joblib"))
    modelo_svm = joblib.load(os.path.join("models", modelo, "svm_mnist.joblib"))
    modelo_cnn = tf.keras.models.load_model(os.path.join("models", modelo, "cnn_mnist.keras"))
    
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
        
        matriz_conf = confusion_matrix(y_teste, preds)
        
        plt.subplot(1, 3, idx)
        sns.heatmap(
            matriz_conf,
            annot=True,
            fmt='d',
            cmap='YlOrBr' if idx == 3 else 'Blues',
            cbar=False,
            xticklabels=range(10),
            yticklabels=range(10)
        )
        plt.title(f"Matriz 10x10: {nome}\nAcurácia: {acc*100:.2f}%", fontweight='bold')
        plt.xlabel("Dígito Predito")
        plt.ylabel("Dígito Real")
    
    plt.tight_layout()
    caminho_grafico = os.path.join("images", modelo, "fase4_matrizes_confusao_comparativas.png")
    plt.savefig(caminho_grafico, dpi=300)
    plt.close()
    
    print(f"Matrizes de Confusão salvas em: '{caminho_grafico}'")
    
    df_metricas = pd.DataFrame(tabela_resultados)
    print("\n=== TABELA COMPARATIVA CONSOLIDADA DE DESEMPENHO ===")
    print(df_metricas.to_string(index=False))
    
    with open(os.path.join("models", modelo, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data_treinamento": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tamanho_teste": len(y_teste),
            "resultados": tabela_resultados,
            "classes": list(range(10))
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Metadados salvos em '{os.path.join('models', modelo, 'metadata.json')}'")

# ==========================================
# EXECUÇÃO DO PIPELINE
# ==========================================

if __name__ == "__main__":
    # Defina a versão aqui: 'V1', 'V2', 'V3' ou a nova 'V4' (Otimizada)
    VERSAO_PIPELINE = 'V4'
    
    # Garantir pastas existentes
    os.makedirs(f"images/{VERSAO_PIPELINE}", exist_ok=True)
    os.makedirs(f"models/{VERSAO_PIPELINE}", exist_ok=True)
    
    # 1. EDA
    X, y = carregar_dataset()
    visualizar_balanceamento_classes(X, y, VERSAO_PIPELINE)
    visualizar_dataset_carregado(X, y, VERSAO_PIPELINE)
    
    # 2. Processamento dos dados de acordo com a versão escolhida
    resultado_split = processamento_e_separacao(X, y, modelo=VERSAO_PIPELINE)
    
    if len(resultado_split) == 6:
        x_treino_cnn, x_teste_cnn, y_treino, y_teste, x_treino_norm, x_teste_norm = resultado_split
    else:
        x_treino_cnn, x_teste_cnn, y_treino, y_teste, x_treino_norm, x_teste_norm = resultado_split
    
    # 3. Treinamento
    tempo_rf, _ = treinar_modelo_RandonForest(x_treino_norm, y_treino, VERSAO_PIPELINE)
    tempo_svm, _ = treinar_modelo_SVM(x_treino_norm, y_treino, VERSAO_PIPELINE)
    tempo_cnn, _ = treinar_modelo_CNN(x_treino_cnn, y_treino, VERSAO_PIPELINE)
    
    tempos = {
        'Random Forest': tempo_rf,
        'SVM(RBF)': tempo_svm,
        'CNN (Deep Learning)': tempo_cnn
    }
    
    # 4. Avaliação e salvamento de metadados
    avaliar_modelos(x_teste_norm, y_teste, x_teste_cnn, tempos, VERSAO_PIPELINE)
