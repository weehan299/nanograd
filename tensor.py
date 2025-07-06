
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()
        self._other = None
        self._out = None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        self._other = other
        self._out = out
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Handle broadcasting for self
                grad = self._unbroadcast(grad, self.data.shape)
                self.grad += grad
                
            if other.requires_grad:
                grad = out.grad
                # Handle broadcasting for other
                grad = self._unbroadcast(grad, other.data.shape)
                other.grad += grad
                
        out._backward = _backward
        out._prev = {self, other}
        return out

    def _unbroadcast(self, grad, original_shape):
        """Reverse broadcasting by summing over broadcasted dimensions"""
        # Handle case where grad has more dimensions than original
        ndims_added = grad.ndim - len(original_shape)
        for i in range(ndims_added):
            grad = grad.sum(axis=0)
        
        # Handle case where dimensions were broadcasted from size 1
        for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
            if orig_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                # Handle broadcasting for self
                grad = self._unbroadcast(grad, self.data.shape)
                self.grad += grad
            if other.requires_grad:
                grad = self.data * out.grad
                # Handle broadcasting for other
                grad = self._unbroadcast(grad, other.data.shape)
                other.grad += grad

        out._backward = _backward
        out._prev = {self, other}
        return out
        
    __rmul__ = __mul__

    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)

    __rsub__ = __sub__

    # Add these methods to your Tensor class in tensor.py

    def __truediv__(self, other):
        """Division operation: self / other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                # d/da (a/b) = 1/b
                grad_a = out.grad / other.data
                # Handle broadcasting
                grad_a = self._unbroadcast(grad_a, self.data.shape)
                self.grad += grad_a
            
            if other.requires_grad:
                # d/db (a/b) = -a/b^2
                grad_b = -out.grad * self.data / (other.data ** 2)
                # Handle broadcasting
                grad_b = self._unbroadcast(grad_b, other.data.shape)
                other.grad += grad_b
        
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rtruediv__(self, other):
        """Reverse division operation: other / self"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self


    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.dot(other.data), requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.dot(other.data.T)
            if other.requires_grad:
                other.grad += self.data.T.dot(out.grad)

        out._backward = _backward
        out._prev = {self, other}
        return out


    def sum(self, dim=None, keepdims=False):
        """
        Sum over given axis (or all axes if dim=None).
        Keeps the result in the autograd graph.
        """
        # Forward pass
        data = self.data.sum(axis=dim, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad)
        # Backward pass
        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad  # Shape of the output
            # If keepdims=True, grad already has the right shape for broadcasting
            if keepdims:
                # grad can be directly broadcast to self.data.shape
                self.grad += grad
            else:
                # We need to restore the summed dimensions for proper broadcasting
                if dim is None:
                    # All dimensions were summed, so grad is a scalar
                    # Broadcast scalar to original shape
                    self.grad += grad
                else:
                    # Handle both single axis and tuple of axes
                    axes = dim if isinstance(dim, tuple) else (dim,)
                    # Normalize negative axes
                    normalized_axes = []
                    for ax in axes:
                        if ax < 0:
                            ax = len(self.data.shape) + ax
                        normalized_axes.append(ax)
                    # Expand dimensions that were summed
                    expanded_grad = grad
                    for ax in sorted(normalized_axes):
                        expanded_grad = np.expand_dims(expanded_grad, axis=ax)
                    
                    self.grad += expanded_grad

        out._backward = _backward
        out._prev = {self}
        return out

    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward = _backward
        out._prev = {self}
        return out

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        # initialize gradient of the output
        self.grad = grad if grad is not None else np.ones_like(self.data)

        # build topological order
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # propagate gradients
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"



# Example usage
if __name__ == "__main__":
    x = Tensor([1, 2, 3], requires_grad=True)
    #y = x * x + 2 * x + 1
    #z = y.sum()
    #z.backward()
    y = Tensor([4,5,6], requires_grad=True)
    z = x*y
    z.backward()

    print("x:", x)
    print("y:", y)
    print("z:", z)
    print("Gradient of x:", x.grad)
    print("Gradient of y:", y.grad)


    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a/b
    c.backward() 
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("gradient of a:", a.grad)
    print("gradient of b:", b.grad)
    print("gradient of c:", c.grad)

    print("*" * 40)
    # Test 1: Sum all elements
    def test_sum_all():
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.sum()
        y.backward()
        print(f"Sum all: {y.data}")  # Should be 10
        print(f"y grad: {y.grad}")  # Should be 1
        print(f"Gradient: {x.grad}")  # Should be [[1, 1], [1, 1]]

    # Test 2: Sum along axis 0
    def test_sum_axis0():
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.sum(dim=0)
        y.backward()
        print(f"Sum axis 0: {y.data}")  # Should be [4, 6]
        print(f"Gradient: {x.grad}")  # Should be [[1, 1], [1, 1]]

    # Test 3: Sum along axis 1
    def test_sum_axis1():
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.sum(dim=1)
        y.backward()
        print(f"Sum axis 1: {y.data}")  # Should be [3, 7]
        print(f"Gradient: {x.grad}")  # Should be [[1, 1], [1, 1]]

    # Test 4: Sum with keepdims=True
    def test_sum_keepdims():
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.sum(dim=0, keepdims=True)
        y.backward()
        print(f"Sum keepdims: {y.data}")  # Should be [[4, 6]]
        print(f"Gradient: {x.grad}")  # Should be [[1, 1], [1, 1]]

    # Test 5: Sum with negative axis
    def test_sum_negative_axis():
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x.sum(dim=-1)  # Same as dim=1
        y.backward()
        print(f"Sum negative axis: {y.data}")  # Should be [3, 7]
        print(f"Gradient: {x.grad}")  # Should be [[1, 1], [1, 1]]

    # Run tests
    test_sum_all()
    test_sum_axis0()
    test_sum_axis1()
    test_sum_keepdims()
    test_sum_negative_axis()