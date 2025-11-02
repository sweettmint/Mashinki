import os
import torch
import random
import numpy as np
import timm
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from config import TrainConfig

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_model(model_name, cfg, device):
    print(f"\n🚀 Обучение модели: {model_name}")
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(cfg.data_dir, transform=transform_train)
    import json, os

    # Сохраняем отображение классов (class -> index)
    with open(os.path.join(os.path.dirname(__file__), "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, ensure_ascii=False, indent=2)
    print("✅ Saved class_to_idx.json:", dataset.class_to_idx)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = timm.create_model(model_name, pretrained=True, num_classes=cfg.num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for epoch in range(cfg.num_epochs):
        losses, preds, labels = [], [], []
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(lbls.cpu().numpy())

        acc = accuracy_score(labels, preds)
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] - Loss: {np.mean(losses):.4f}, Accuracy: {acc:.4f}")

    return model, acc


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_train = ["resnet18", "efficientnet_b0"]
    results = {}

    for model_name in models_to_train:
        model, acc = train_one_model(model_name, cfg, device)
        results[model_name] = acc
        torch.save(model.state_dict(), f"{model_name}.pth")

    # Выбор лучшей модели
    best_model_name = max(results, key=results.get)
    print(f"\n🏆 Лучшая модель: {best_model_name} (Accuracy={results[best_model_name]:.4f})")

    # Экспорт в ONNX
    best_model = timm.create_model(best_model_name, pretrained=False, num_classes=cfg.num_classes)
    best_model.load_state_dict(torch.load(f"{best_model_name}.pth", map_location=device))
    best_model.to(device)
    best_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(best_model, dummy_input, cfg.save_path,
                      input_names=['input'], output_names=['output'], opset_version=16)
    print(f"✅ Модель экспортирована в {cfg.save_path}")

if __name__ == "__main__":
    main()

