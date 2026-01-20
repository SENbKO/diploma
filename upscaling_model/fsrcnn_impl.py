import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from FSRCNN import FSRCNN
from dataset_loader import SRDataset
from tqdm import tqdm
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Set up model and its training parameters
    model = FSRCNN(scale_factor=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dataset_folder = Path(__file__).parent / ".." / "dataset" / "DIV2K_train_HR" / "DIV2K_train_HR"
    dataset = SRDataset(dataset_folder, scale=2, crop_size=96)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for lr, hr in loop:
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(loader):.6f}")

    model.eval()
    with torch.no_grad():
        lr, _ = dataset[0]
        lr = lr.unsqueeze(0).to(device)
        sr = model(lr)

    torch.save(model.state_dict(), "fsrcnn_x2.pth")

if __name__ == "__main__":
    main()



