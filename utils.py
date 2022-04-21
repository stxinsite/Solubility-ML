import torch
DEVICE = torch.device("cuda")


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    num = 0
    for data in loader:
        #print("Batch", num, data)
        dev_data = data.to(device)
        # Compute prediction error
        pred = model(dev_data)#.reshape(dev_y.size())
        #print(pred.size(), dev_data.y.size())
        loss = loss_fn(pred, dev_data.y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num += 1
    return loss.item()
        
        
def test(model, dataset, loss_fn):
    model.eval()
    diff = 0
    counter = 0
    with torch.no_grad():
        for data in dataset:
            data_x = data.to(DEVICE)
            pred = model(data_x) #.reshape(dev_data.y.size())
            diff += loss_fn(pred, data.y.to(DEVICE)).item()
            counter += 1
    return diff/counter