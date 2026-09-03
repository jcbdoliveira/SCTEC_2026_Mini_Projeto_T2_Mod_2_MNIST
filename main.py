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
    processamento_e_separacao(dadosX, rotulosY)
