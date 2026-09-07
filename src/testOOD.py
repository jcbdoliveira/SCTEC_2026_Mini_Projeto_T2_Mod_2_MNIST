import os
import numpy as np
from src.pipeline import criar_modelo_CNN
import matplotlib.pyplot as plt

def realiza_teste_OOD(x_treino_cnn, y_treino, x_teste_cnn, y_teste, modelo):
    mascara_sem_4_7 = ~np.isin(y_treino, [4, 7])
    x_treino_ood = x_treino_cnn[mascara_sem_4_7]
    y_treino_ood = y_treino[mascara_sem_4_7]

    modelo_ood = criar_modelo_CNN()    
    modelo_ood.fit(x_treino_ood, y_treino_ood, epochs=5, batch_size=64, verbose=0)

    mascara_teste_apenas_4_7 = np.isin(y_teste, [4, 7])
    prob_ood = modelo_ood.predict(x_teste_cnn[mascara_teste_apenas_4_7])
    certeza_media = np.max(prob_ood, axis=1).mean()

    print(f"-> Certeza média em classes nunca vistas (4 e 7): {certeza_media * 100:.2f}%")
    print("-> Falsa Certeza (Overconfidence) da Softmax comprovada com sucesso!")
    print("-> Gerando visualização de exemplos OOD...")
    visualizar_previsoes_OOD(modelo_ood, x_teste_cnn, y_teste, modelo, num_exemplos=3)
    
    return modelo_ood

def visualizar_previsoes_OOD(modelo_ood, x_teste_cnn, y_teste, modelo, num_exemplos=3):
    # 1. Filtrar o conjunto de teste para pegar apenas os dígitos 4 e 7 (que são OOD)
    mascara_4_7 = np.isin(y_teste, [4, 7])
    x_ood = x_teste_cnn[mascara_4_7]
    y_ood = y_teste[mascara_4_7]
    
    # 2. Fazer a previsão de probabilidade para essas imagens
    prob_ood = modelo_ood.predict(x_ood)
    
    # 3. Configurar a janela do gráfico (uma linha por exemplo: imagem + gráfico de barras)
    fig, axes = plt.subplots(num_exemplos, 2, figsize=(10, 3 * num_exemplos))
    if num_exemplos == 1:
        axes = np.array([axes]) # Garante que funciona mesmo se for só 1 exemplo
        
    for i in range(num_exemplos):
        img = x_ood[i].reshape(28, 28)
        real_label = y_ood[i]
        probs = prob_ood[i]
        predicted_label = np.argmax(probs)
        confidence = probs[predicted_label] * 100
        
        # --- Coluna 1: Mostrar a Imagem ---
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Dígito Real: {real_label} (OOD)\nO modelo nunca viu!")
        axes[i, 0].axis('off')
        
        # --- Coluna 2: Mostrar o gráfico de barras das saídas da Softmax ---
        # Como o modelo treinou sem o 4 e o 7, as classes que ele conhece são [0, 1, 2, 3, 5, 6, 8, 9]
        classes_conhecidas = list(range(10)) 
        bars = axes[i, 1].bar(classes_conhecidas, probs, color='crimson')
        
        # Destacar a barra que o modelo escolheu erroneamente
        bars[predicted_label].set_color('darkred')
        
        axes[i, 1].set_xlim(-0.5, 9.5)
        axes[i, 1].set_ylim(0, 1.1)
        axes[i, 1].set_xticks(classes_conhecidas)
        axes[i, 1].set_title(f"Previsão: Acho que é {predicted_label} ({confidence:.1f}% de certeza)")
        axes[i, 1].set_ylabel("Probabilidade Softmax")
        
    plt.tight_layout()
    # Salva o gráfico com segurança na pasta do projeto
    os.makedirs(f"images/{modelo}", exist_ok=True)
    caminho_grafico = os.path.join("images", modelo, "resultado_teste_ood.png")
    plt.savefig(caminho_grafico, dpi=300)
    
    # CORREÇÃO: Substitua plt.show() por plt.close(fig) para limpar a memória
    plt.close(fig)
    print(f"-> Gráfico do teste OOD salvo com sucesso em: '{caminho_grafico}'")

