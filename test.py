


from nn import mse_loss,MLP
from tensor import Tensor
import numpy as np
import optimizer

if __name__ == "__main__":
    #test_scalar_ops()
    print("-" * 40)
    # Synthetic: y = 2x + 3 + noise
    np.random.seed(42)
    X_np = np.random.rand(200, 1)
    y_np = 2 * X_np + + 0.1 * np.random.randn(200, 1)

    X = Tensor(X_np, requires_grad=False)
    y = Tensor(y_np, requires_grad=False)

    model = MLP(in_dim=1, hidden_dim=32, out_dim=1)
    #print(len(list(model.parameters())))
    print(sum(p.data.size for p in model.parameters()))
    optim = optimizer.Adam(model.parameters(), lr=0.005)

    #lr = 0.005
    epochs = 1 

    for epoch in range(1, epochs+1):
        preds = model(X)           # calls MLP.forward via Module.__call__
        loss  = mse_loss(preds, y)

        model.zero_grad()
        loss.backward()
        optim.step()          # calls Adam.step()

        # SGD step
        #for p in model.parameters():
        #    p.data -= lr * p.grad

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss = {loss.data:.4f}")
