import torch
from tqdm import tqdm

def inference(model, test_loader, device, val=False):
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        if val:
            for features in tqdm(iter(test_loader)):
                features = features[0].float().to(device)
                
                probs = model(features)

                probs  = probs.cpu().detach().numpy()
                predictions += probs.tolist()

        else :
            for features in tqdm(iter(test_loader)):
                features = features.float().to(device)
                
                probs = model(features)

                probs  = probs.cpu().detach().numpy()
                predictions += probs.tolist()
        
    return predictions