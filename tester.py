import torch
from torch.utils.data import DataLoader
from dataset import FireDataset
from model import FireTransformer
from config import Config
from model import FireTransformer
import matplotlib.pyplot as plt
from pathlib import Path

def test():
    '''
    Function to test the model
    '''

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    dataset = FireDataset(Config.SEQUENCE_LENGTH, istrain=False)
    dataset = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Load the model
    model = FireTransformer(Config).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pt', weights_only = True))
    criterion = torch.nn.MSELoss()
    
    # Validation
    model.eval()
    val_loss = 0
    
    # if test results folder does not exist, create it
    Path('test_results').mkdir(exist_ok=True)
    
    # random visualization
    img_1 = torch.tensor([0.0]*50*50).view(50, 50)
    
    # plot
    plt.figure(figsize=(10, 5))
    plt.imshow(img_1, cmap='YlOrRd')
    plt.title('Predicted Fire Spread')
    plt.tight_layout()
    plt.savefig('test_results/random_visualization.png')
    
    # INFERENCE
    with torch.no_grad():

        # Iterate over the dataset
        for i, (data, target) in enumerate(dataset):
            data, target = data.to(device), target.to(device)
            bs = data.size(0)
            output = model(data)
            
            # Visualize the first prediction for each batch
            for j in range(bs):
                
                # Visualize the first prediction
                first_img = output[j]
                first_img = torch.where(first_img < 0.5, 0, 255)
                first_img = first_img.cpu().detach().numpy()
                first_img = first_img.reshape((50, 50))
                
                # Visualize the first target
                first_target = target[j].cpu().detach().numpy()
                first_target = first_target.reshape((50, 50))
                
                # Plot
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(first_img, cmap='YlOrRd')
                plt.title('Predicted Fire Spread')
                plt.subplot(1, 2, 2)
                plt.imshow(first_target, cmap='YlOrRd')
                plt.title('True Fire Spread')
                plt.tight_layout()
                plt.savefig(f'test_results/first_prediction-{i}-{j}.png')
                plt.close()
            
            # Compute loss
            val_loss += criterion(output, target).item()
    
    # Print final validation loss
    val_loss /= len(dataset)
    print(f' Val Loss: {val_loss:.6f}')

if __name__ == "__main__":
    test()
