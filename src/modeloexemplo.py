#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script utilitário para gerar o arquivo 'models/V2/cnn_mnist.keras'
caso você ainda não tenha treinado ou queira testar a visualização rapidamente.
"""

import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("Gerador de Modelo CNN MNIST para Teste do TensorSpace")
    print("=" * 60)
    
    try:
        import tensorflow as tf
    except ImportError:
        print("[ERRO] TensorFlow não instalado. Execute: pip install tensorflow")
        sys.exit(1)

    target_dir = Path("models/V2")
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "cnn_mnist.keras"

    print(f"Construindo arquitetura LeNet-5 adaptada para MNIST...")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1), name="mnist_input"),
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding="same", activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_1"),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding="valid", activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(120, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(84, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(10, activation="softmax", name="output_digits"),
    ], name="cnn_mnist")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("\nResumo do modelo gerado:")
    model.summary()

    # Salvar no formato moderno Keras v3 (.keras)
    model.save(model_path)
    print(f"\n[SUCESSO] Modelo salvo em: {model_path.resolve()}")
    print("Agora você pode executar: python visualizar_modelo.py")

if __name__ == "__main__":
    main()
