# Tutorial of Transfer learning on fashion mnist
from torchvision import models
from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np

n_epochs = 10

class FashionResnet(torch.nn.Module):
    def __init__(self):
        super(FashionResnet, self).__init__()
        #model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.rn= models.resnet50(weights=models.ResNet50_Weights.DEFAULT)# pretrained=True,
        self.rn.fc = torch.nn.Linear(self.rn.fc.in_features,128)
        self.fc2 = torch.nn.Linear(self.rn.fc.out_features,10)
    def forward(self, x):
        with torch.no_grad():
            x = self.rn(x)
        x = self.fc2(x)
        return x

frn = FashionResnet()

normalize = transforms.Normalize(mean=[x/255 for x in [125.3, 123.0,113.9]],std=[x/255 for x in [63.0, 62.1,66.7]])

mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),normalize])

mnist_train = datasets.FashionMNIST('.', train=True, download=True,transform=mnist_transforms)
mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

mnist_test = datasets.FashionMNIST('.', train=False, download=True,transform=mnist_transforms)
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

model_path = Path("model.pt")
if model_path.exists():
    frn.load_state_dict(torch.load(model_path))
    print("Loaded ok")


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
optimizer = torch.optim.Adam(frn.parameters(),lr=0.0001)
best_loss=1e14
training_loss_evolution=[]
test_loss_evolution=[]
for epoch in range(n_epochs):
    print(f"Epoch : {epoch}/{n_epochs}")
    losses=[]
    for i, batch in enumerate(tqdm(mnist_train_loader)) :
        x,y=batch
        y_pred = frn(x)
        loss = torch.nn.functional.cross_entropy(y_pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    mean_loss = np.mean(np.array(losses))
    print(f"Training Loss : {mean_loss}")
    training_loss_evolution.append(mean_loss)

    losses=[]
    # Validate
    for i, batch in enumerate(tqdm(mnist_test_loader)) :
        x,y=batch
        y_pred = frn(x)
        loss = torch.nn.functional.cross_entropy(y_pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    mean_loss = np.mean(np.array(losses))
    print(f"Test Loss : {mean_loss}")
    test_loss_evolution.append(mean_loss)
    if mean_loss<best_loss:
        best_loss=mean_loss
        torch.save(frn.state_dict(),str(model_path))
        print("Saved")

torch.save(frn.state_dict(),str(model_path))


plt.figure()
plt.plot(training_loss_evolution,label="Training loss")
plt.plot(test_loss_evolution,label="Testing loss")
plt.title("Loss evolution")

print("Testing")
x,y=next(mnist_test_loader)
print(f" {i} : {x.shape} {y.shape}")
y_true = y.numpy()
with torch.no_grad():
    preds = frn(x)
    y_pred = preds.numpy().argmax(axis=1)
plt.figure()
for j in range(32):
    plt.subplot(4,8,j+1)
    print(labels_map[y.numpy()[j]])
    plt.imshow(x[j].numpy().transpose(1,2,0))
    plt.title(f"{labels_map[y_true[j]]} / {labels_map[y_pred[j]]}")
plt.tight_layout()
plt.show()

