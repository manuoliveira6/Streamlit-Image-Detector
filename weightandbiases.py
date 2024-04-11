import argparse
import wandb
import torchvision
from cnn import CNN, load_data
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os


classification_models = torchvision.models.list_models(module=torchvision.models)

def validate_model(model_name):
    if model_name not in classification_models:
        raise ValueError(f"El modelo {model_name} no está en la lista de modelos de clasificación disponibles.")

def main(model, learning_rate, epochs):
    # Validar el modelo
    validate_model(model)

    wandb.init(
        project="Streamlit",
        config={
            "learning_rate": learning_rate,
            "architecture": "my_model",
            "dataset": "my_dataset",
            "epochs": epochs,
        }
    )

    train_dir = '../dataset/training'
    valid_dir = '../dataset/validation'

    train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=32, 
                                                    img_size=224) 

    my_model = CNN(getattr(torchvision.models, model)(weights='DEFAULT'), num_classes)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print('Empezando el entrenamiento...')
    for epoch in range(epochs):
        # Entrenamiento
        my_model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for images, labels in train_loader:
            optimizer.zero_grad()
            output = my_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(output, 1).tolist())
            train_targets.extend(labels.tolist())

        # Validación
        my_model.eval()
        valid_losses = []
        valid_preds = []
        valid_targets = []

        with torch.no_grad():
            for images, labels in valid_loader:
                output = my_model(images)
                loss = criterion(output, labels)

                valid_losses.append(loss.item())
                valid_preds.extend(torch.argmax(output, 1).tolist())
                valid_targets.extend(labels.tolist())

        # Calcular métricas
        train_accuracy = accuracy_score(train_targets, train_preds)
        valid_accuracy = accuracy_score(valid_targets, valid_preds)

        # Registrar métricas en W&B
        wandb.log({"train_loss": sum(train_losses) / len(train_losses),
                   "train_accuracy": train_accuracy,
                   "valid_loss": sum(valid_losses) / len(valid_losses),
                   "valid_accuracy": valid_accuracy})
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {sum(train_losses) / len(train_losses)}, Train Accuracy: {train_accuracy}, Valid Loss: {sum(valid_losses) / len(valid_losses)}, Valid Accuracy: {valid_accuracy}")

    if not os.path.exists('models'):
        os.makedirs('models')
    my_model.save(f'models/{model}-{epochs}epoch-{learning_rate}lr')


# Configurar y parsear los argumentos de línea de comandos
parser = argparse.ArgumentParser(description="Script para entrenar modelos de clasificación.")
parser.add_argument("--model", type=str, default='resnet50', nargs='?', help="Nombre del modelo de clasificación (ej. ResNet50)")
parser.add_argument("--learning_rate", type=float, default=0.001, nargs='?', help="Tasa de aprendizaje (default: 0.001)")
parser.add_argument("--epochs", type=int, default=10, nargs='?', help="Número de epochs (default: 10)")
args = parser.parse_args()

if __name__ == "__main__":
    main(args.model, args.learning_rate, args.epochs)
    # python weightandbiases.py --model resnet50 --learning_rate 0.001 --epochs 10
