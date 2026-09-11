import sys
from src.visualizador import main
from src.modeloexemplo import main as geraModeloExemplo
from src.install import verificar_e_instalar_bibliotecas, verifica_pastas

modelo = sys.argv[1] if len(sys.argv) > 1 else 'V2'

verifica_pastas(modelo)
verificar_e_instalar_bibliotecas()

if modelo == "EX":
    geraModeloExemplo()
    modelo = "V2"

main(modelo) 
