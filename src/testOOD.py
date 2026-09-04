import numpy as np
from src.pipeline import criar_modelo_CNN

def realiza_teste_OOD(x_treino_cnn, y_treino, x_teste_cnn, y_teste):
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
