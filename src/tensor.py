
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
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
        out = Tensor(self.data @ other.data,
                    requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                # swap last two dims of other.data
                b_T = other.data.swapaxes(-1, -2)
                grad_a = out.grad @ b_T
                # handle broadcasting back to self.data.shape if needed
                grad_a = self._unbroadcast(grad_a, self.data.shape)
                self.grad += grad_a

            if other.requires_grad:
                # swap last two dims of self.data
                a_T = self.data.swapaxes(-1, -2)
                grad_b = a_T @ out.grad
                # if other.data was broadcast, unbroadcast it
                grad_b = other._unbroadcast(grad_b, other.data.shape)
                other.grad += grad_b

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

    def masked_fill(self, mask: np.ndarray, value: float) -> "Tensor":
        """
        Return a new Tensor where entries corresponding to True in the mask
        are set to `value`, with gradients blocked through those positions.
        `mask` should be a boolean array broadcastable to self.data.shape.
        """
        # Ensure mask is a boolean array
        mask = np.asarray(mask, dtype=bool)
        
        # Forward pass - handle broadcasting properly
        out_data = self.data.copy()
        
        # Use numpy's where function which handles broadcasting properly
        # First, we need to broadcast the mask to match self.data.shape
        try:
            # Try to broadcast mask to self.data.shape
            broadcasted_mask = np.broadcast_to(mask, self.data.shape)
            out_data = np.where(broadcasted_mask, value, out_data)
        except ValueError as e:
            # If broadcasting fails, provide a helpful error message
            raise ValueError(f"Cannot broadcast mask with shape {mask.shape} to tensor with shape {self.data.shape}. "
                            f"Original error: {e}")
        
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                # upstream grad, but zeroed where mask is True
                # Use the same broadcasting logic as the forward pass
                try:
                    broadcasted_mask = np.broadcast_to(mask, self.data.shape)
                    grad_mask = np.where(broadcasted_mask, 0.0, out.grad)
                    self.grad += grad_mask
                except ValueError:
                    # This should not happen if forward pass succeeded
                    pass

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

    def transpose(self, axis1=-2, axis2=-1):
        """Transpose tensor along specified axes"""
        # Handle negative axes
        if axis1 < 0:
            axis1 = self.data.ndim + axis1
        if axis2 < 0:
            axis2 = self.data.ndim + axis2
        
        # Create axes permutation
        axes = list(range(self.data.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        
        out_data = np.transpose(self.data, axes)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                # Transpose gradient back
                self.grad += np.transpose(out.grad, axes)
        
        out._backward = _backward
        out._prev = {self}
        return out
        
    def reshape(self, new_shape):
        """Reshape tensor to new shape"""
        out_data = self.data.reshape(new_shape)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        out._prev = {self}
        return out

            
    def backward(self, grad=None):
        """Optimized backward pass with memory management"""
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        
        self.grad = grad if grad is not None else np.ones_like(self.data)

        # OPTIMIZATION: Use iterative approach instead of recursive for large graphs
        topo = []
        visited = set()
        stack = [self]
        
        # Build topological order iteratively
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            # Check if all children have been visited
            all_children_visited = all(child in visited for child in node._prev)
            
            if all_children_visited:
                visited.add(node)
                topo.append(node)
            else:
                # Add back to stack and add unvisited children
                stack.append(node)
                for child in node._prev:
                    if child not in visited:
                        stack.append(child)

        # Propagate gradients
        for node in reversed(topo):
            node._backward()
            
        # OPTIMIZATION: Clear computation graph after backward pass to save memory
        self._clear_graph()

    def _clear_graph(self):
        """Clear computation graph to save memory"""
        visited = set()
        stack = [self]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            
            # Clear backward function and previous nodes
            node._backward = lambda: None
            for child in node._prev:
                stack.append(child)
            node._prev.clear()

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

