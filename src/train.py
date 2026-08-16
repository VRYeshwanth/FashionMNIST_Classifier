import torch

def train_step(model, dataloader, loss_fn, optimizer, metrics, device):
    model.train()

    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        for metric in metrics.values():
            metric.update(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(
                f"Looked at {batch * len(X)} / "
                f"{len(dataloader.dataset)} samples"
            )

    train_loss /= len(dataloader)

    results = {
        "loss": train_loss
    }

    for name, metric in metrics.items():
        results[name] = metric.compute().item()
        metric.reset()

    return results


def test_step(model, dataloader, loss_fn, metrics, device):
    model.eval()

    test_loss = 0

    with torch.inference_mode():

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            test_loss += loss_fn(y_pred, y).item()

            for metric in metrics.values():
                metric.update(y_pred, y)

    test_loss /= len(dataloader)

    results = {
        "loss": test_loss
    }

    for name, metric in metrics.items():
        results[name] = metric.compute().item()
        metric.reset()

    return results