from src.install import verificar_e_instalar_bibliotecas

verificar_e_instalar_bibliotecas()

if __name__ == "__main__":
    from src.pipeline import carregar_dataset, visualizar_dataset_carregado

    carregar_dataset()
    #visualizar_dataset_carregado()