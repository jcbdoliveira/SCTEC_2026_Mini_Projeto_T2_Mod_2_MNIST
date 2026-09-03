from src.install import verificar_e_instalar_bibliotecas, verifica_pastas

verifica_pastas()
verificar_e_instalar_bibliotecas()

if __name__ == "__main__":
    from src.pipeline import *

    #Etapa 01 
    dadosX, rotulosY = carregar_dataset()
    visualizar_dataset_carregado(dadosX, rotulosY)
    visualizar_balanceamento_classes(dadosX, rotulosY)

    #Etapa 02
    x_treino_cnn, x_teste_cnn, y_treino, y_teste, x_treino_norm, x_teste_norm = processamento_e_separacao(dadosX, rotulosY)

    #Etapa 03
    tempos_treino = {}
    modelos_salvos = {}

    tempo, pasta = treinar_modelo_RandonForest(x_treino_norm, y_treino)
    tempos_treino['Random Forest'] = tempo 
    #------------------------------------------------------------------
    tempo, pasta = treinar_modelo_SVM(x_treino_norm, y_treino)
    tempos_treino['SVM(RBF)'] = tempo
    #------------------------------------------------------------------
    tempo, pasta = treinar_modelo_CNN(x_treino_cnn, y_treino)
    tempos_treino['CNN (Deep Learning)'] = tempo
