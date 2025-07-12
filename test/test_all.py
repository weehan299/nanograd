import unittest
import numpy as np
import tempfile
import os
from src.tensor import Tensor
from src.module import Module
from src.nn import Linear, ReLU, MLP, Embedding, PositionalEncoding
from src.dropout import Dropout
from src.layernorm import LayerNorm
from src.multihead_attention import Head, MultiHeadAttention, FeedForward, Block
from src.gptlanguage import GPTLanguageModel
from src.optimizer import Adam
from src.softmax import softmax, exp, log, cross_entropy
from src.model_state import ModelState, save_model, load_model, model_exists


class TestTensor(unittest.TestCase):
    """Test the Tensor class and its operations"""
    
    def test_tensor_creation(self):
        """Test tensor creation with different data types"""
        # Test with list
        t1 = Tensor([1, 2, 3])
        self.assertEqual(t1.data.shape, (3,))
        self.assertEqual(t1.data.dtype, np.float32)
        
        # Test with numpy array
        t2 = Tensor(np.array([[1, 2], [3, 4]]))
        self.assertEqual(t2.data.shape, (2, 2))
        
        # Test requires_grad
        t3 = Tensor([1, 2], requires_grad=True)
        self.assertTrue(t3.requires_grad)
        self.assertIsNotNone(t3.grad)
        
    def test_addition(self):
        """Test tensor addition and broadcasting"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        
        c = a + b
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad, np.ones_like(b.data))
        
    def test_broadcasting_addition(self):
        """Test addition with broadcasting"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([10, 20], requires_grad=True)
        
        c = a + b
        expected = np.array([[11, 22], [13, 24]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient with broadcasting
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad, np.array([2, 2]))  # Sum over broadcast dims
        
    def test_multiplication(self):
        """Test tensor multiplication"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[2, 3], [4, 5]], requires_grad=True)
        
        c = a * b
        expected = np.array([[2, 6], [12, 20]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, b.data)
        np.testing.assert_array_equal(b.grad, a.data)
        
    def test_division(self):
        """Test tensor division"""
        a = Tensor([[6, 8], [10, 12]], requires_grad=True)
        b = Tensor([[2, 4], [5, 6]], requires_grad=True)
        
        c = a / b
        expected = np.array([[3, 2], [2, 2]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, 1.0 / b.data)
        expected_b_grad = -a.data / (b.data ** 2)
        np.testing.assert_array_almost_equal(b.grad, expected_b_grad)
        
    def test_matmul(self):
        """Test matrix multiplication"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        
        c = a.matmul(b)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, np.ones_like(c.data) @ b.data.T)
        np.testing.assert_array_equal(b.grad, a.data.T @ np.ones_like(c.data))
        
    def test_sum(self):
        """Test sum operation"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        
        # Sum all elements
        c = a.sum()
        self.assertEqual(c.data, 10)
        
        c.backward()
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        
        # Sum along axis
        a.grad = np.zeros_like(a.data)  # Reset gradient
        c = a.sum(dim=0)
        np.testing.assert_array_equal(c.data, [4, 6])
        
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        
    def test_masked_fill(self):
        """Test masked fill operation"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        mask = np.array([[True, False], [False, True]])
        
        c = a.masked_fill(mask, -999)
        expected = np.array([[-999, 2], [3, -999]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient (should be zero where mask is True)
        c.backward(np.ones_like(c.data))
        expected_grad = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(a.grad, expected_grad)
        
    def test_transpose(self):
        """Test transpose operation"""
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        
        c = a.transpose()
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
        
    def test_reshape(self):
        """Test reshape operation"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        
        c = a.reshape((4,))
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test gradient
        c.backward(np.ones_like(c.data))
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))


class TestModule(unittest.TestCase):
    """Test the Module base class"""
    
    def test_parameter_registration(self):
        """Test automatic parameter registration"""
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Tensor([[1, 2], [3, 4]], requires_grad=True)
                self.bias = Tensor([1, 2], requires_grad=True)
                
        module = TestModule()
        params = list(module.parameters())
        self.assertEqual(len(params), 2)
        self.assertIn(module.weight, params)
        self.assertIn(module.bias, params)
        
    def test_submodule_registration(self):
        """Test automatic submodule registration"""
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(2, 3)
                
        module = TestModule()
        params = list(module.parameters())
        # Should include parameters from submodule
        self.assertGreater(len(params), 0)
        
    def test_zero_grad(self):
        """Test zero_grad functionality"""
        module = Linear(2, 3)
        # Set some gradients
        for param in module.parameters():
            param.grad = np.ones_like(param.data)
            
        module.zero_grad()
        
        # Check all gradients are zero
        for param in module.parameters():
            np.testing.assert_array_equal(param.grad, np.zeros_like(param.data))


class TestLinear(unittest.TestCase):
    """Test the Linear layer"""
    
    def test_forward_pass(self):
        """Test forward pass of linear layer"""
        linear = Linear(3, 2)
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        
        y = linear(x)
        self.assertEqual(y.data.shape, (2, 2))
        
    def test_backward_pass(self):
        """Test backward pass of linear layer"""
        linear = Linear(2, 3)
        x = Tensor([[1, 2]], requires_grad=True)
        
        y = linear(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(linear.weight.grad)
        if linear.bias is not None:
            self.assertIsNotNone(linear.bias.grad)
            
    def test_no_bias(self):
        """Test linear layer without bias"""
        linear = Linear(2, 3, bias=False)
        self.assertIsNone(linear.bias)
        
        x = Tensor([[1, 2]])
        y = linear(x)
        self.assertEqual(y.data.shape, (1, 3))


class TestReLU(unittest.TestCase):
    """Test the ReLU activation function"""
    
    def test_forward_pass(self):
        """Test ReLU forward pass"""
        relu = ReLU()
        x = Tensor([[-1, 0, 1, 2]], requires_grad=True)
        
        y = relu(x)
        expected = np.array([[0, 0, 1, 2]])
        np.testing.assert_array_equal(y.data, expected)
        
    def test_backward_pass(self):
        """Test ReLU backward pass"""
        relu = ReLU()
        x = Tensor([[-1, 0, 1, 2]], requires_grad=True)
        
        y = relu(x)
        y.backward(np.ones_like(y.data))
        
        expected_grad = np.array([[0, 0, 1, 1]])
        np.testing.assert_array_equal(x.grad, expected_grad)


class TestEmbedding(unittest.TestCase):
    """Test the Embedding layer"""
    
    def test_forward_pass(self):
        """Test embedding forward pass"""
        embedding = Embedding(10, 5)  # 10 tokens, 5 dimensions
        indices = Tensor([[0, 1, 2], [3, 4, 5]])
        
        output = embedding(indices)
        self.assertEqual(output.data.shape, (2, 3, 5))
        
    def test_backward_pass(self):
        """Test embedding backward pass"""
        embedding = Embedding(5, 3)
        indices = Tensor([[0, 1, 2]])
        
        output = embedding(indices)
        loss = output.sum()
        loss.backward()
        
        # Check weight gradients
        self.assertIsNotNone(embedding.weight.grad)
        
    def test_out_of_bounds_error(self):
        """Test error handling for out of bounds indices"""
        embedding = Embedding(5, 3)
        indices = Tensor([[0, 1, 5]])  # 5 is out of bounds
        
        with self.assertRaises(ValueError):
            embedding(indices)


class TestDropout(unittest.TestCase):
    """Test the Dropout layer"""
    
    def test_training_mode(self):
        """Test dropout in training mode"""
        dropout = Dropout(p=0.5)
        dropout.training = True
        
        x = Tensor(np.ones((100, 10)), requires_grad=True)
        y = dropout(x)
        
        # Should have some zeros and some scaled values
        self.assertLess(np.sum(y.data == 0), x.data.size)  # Not all zeros
        self.assertGreater(np.sum(y.data == 0), 0)  # Some zeros
        
    def test_inference_mode(self):
        """Test dropout in inference mode"""
        dropout = Dropout(p=0.5)
        dropout.training = False
        
        x = Tensor(np.ones((10, 10)))
        y = dropout(x)
        
        # Should be identical to input
        np.testing.assert_array_equal(x.data, y.data)
        
    def test_edge_cases(self):
        """Test dropout edge cases"""
        # p=0 should return input unchanged
        dropout = Dropout(p=0.0)
        x = Tensor(np.ones((5, 5)), requires_grad=True)
        y = dropout(x)
        np.testing.assert_array_equal(x.data, y.data)
        
        # p=1 should return zeros
        dropout = Dropout(p=1.0)
        x = Tensor(np.ones((5, 5)), requires_grad=True)
        y = dropout(x)
        np.testing.assert_array_equal(y.data, np.zeros_like(x.data))
        
    def test_invalid_probability(self):
        """Test error handling for invalid dropout probability"""
        with self.assertRaises(ValueError):
            Dropout(p=-0.1)
            
        with self.assertRaises(ValueError):
            Dropout(p=1.5)


class TestLayerNorm(unittest.TestCase):
    def test_forward_pass_basic(self):
        ln = LayerNorm(10)
        x = Tensor(np.random.randn(2, 10), requires_grad=True)
        y = ln(x)
        # shape
        self.assertEqual(y.data.shape, x.data.shape)
        # mean≈0, std≈1
        np.testing.assert_allclose(y.data.mean(axis=-1, keepdims=True),0, atol=1e-5)
        np.testing.assert_allclose(y.data.std(), 1, atol=1e-4)

    def test_forward_pass_3d(self):
        ln = LayerNorm((4,5))
        x = Tensor(np.random.randn(2,4,5), requires_grad=True)
        y = ln(x)
        self.assertEqual(y.data.shape, x.data.shape)
        np.testing.assert_allclose(
            y.data.mean(axis=(1,2), keepdims=True),
            0, atol=1e-5
        )
        np.testing.assert_allclose(
            y.data.std(axis=(1,2), keepdims=True),
            1, atol=1e-5
        )

    def test_forward_pass_constant(self):
        ln = LayerNorm(3)
        x = Tensor(np.ones((2,3)), requires_grad=True)
        y = ln(x)
        # all zeros
        np.testing.assert_allclose(y.data, 0, atol=1e-6)

    def test_backward_with_affine(self):
        ln = LayerNorm(3)
        x = Tensor([[1.,2.,3.]], requires_grad=True)
        y = ln(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ln.weight.grad)
        self.assertIsNotNone(ln.bias.grad)
        # optional: check shapes
        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(ln.weight.grad.shape, (3,))
        self.assertEqual(ln.bias.grad.shape, (3,))

    def test_backward_without_affine(self):
        ln = LayerNorm(3, elementwise_affine=False)
        x = Tensor([[1.,2.,3.]], requires_grad=True)
        y = ln(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # weight/bias should remain None
        self.assertIsNone(ln.weight)
        self.assertIsNone(ln.bias)

    def test_shape_mismatch_raises(self):
        ln = LayerNorm(4)
        with self.assertRaises(ValueError):
            ln(Tensor(np.random.randn(2,3)))

class TestSoftmax(unittest.TestCase):
    """Test softmax and related functions"""
    
    def test_softmax_forward(self):
        """Test softmax forward pass"""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        y = softmax(x, dim=-1)
        
        # Check probabilities sum to 1
        sums = np.sum(y.data, axis=-1)
        np.testing.assert_array_almost_equal(sums, [1, 1])
        
        # Check all values are positive
        self.assertTrue(np.all(y.data > 0))
        
    def test_softmax_backward(self):
        """Test softmax backward pass"""
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = softmax(x, dim=-1)
        loss = y.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        
    def test_exp_function(self):
        """Test exponential function"""
        x = Tensor([0, 1, 2], requires_grad=True)
        y = exp(x)
        
        expected = np.exp([0, 1, 2])
        np.testing.assert_array_almost_equal(y.data, expected)
        
        # Test gradient
        y.backward(np.ones_like(y.data))
        np.testing.assert_array_almost_equal(x.grad, expected)  # d/dx exp(x) = exp(x)
        
    def test_log_function(self):
        """Test logarithm function"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = log(x)
        
        expected = np.log([1, 2, 3])
        np.testing.assert_array_almost_equal(y.data, expected)
        
        # Test gradient
        y.backward(np.ones_like(y.data))
        expected_grad = 1.0 / x.data
        np.testing.assert_array_almost_equal(x.grad, expected_grad)
        
    def test_cross_entropy(self):
        """Test cross entropy loss"""
        logits = Tensor([[2, 1, 0], [1, 3, 2]], requires_grad=True)
        targets = Tensor([0, 1])
        
        loss = cross_entropy(logits, targets)
        
        # Loss should be positive
        self.assertGreater(loss.data.item(), 0)
        
        # Test gradient
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention components"""
    
    def test_head_forward(self):
        """Test single attention head"""
        head = Head(head_size=8, n_embd=16, block_size=10)
        x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)
        
        y = head(x)
        self.assertEqual(y.data.shape, (2, 5, 8))
        
    def test_multihead_attention(self):
        """Test multi-head attention"""
        mha = MultiHeadAttention(num_heads=4, head_size=8, n_embd=32, block_size=10)
        x = Tensor(np.random.randn(2, 5, 32), requires_grad=True)
        
        y = mha(x)
        self.assertEqual(y.data.shape, (2, 5, 32))
        
    def test_feedforward(self):
        """Test feed-forward network"""
        ff = FeedForward(n_embd=16)
        x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)
        
        y = ff(x)
        self.assertEqual(y.data.shape, (2, 5, 16))
        
    def test_transformer_block(self):
        """Test transformer block"""
        block = Block(n_embd=16, n_head=4, block_size=10)
        x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)
        
        y = block(x)
        self.assertEqual(y.data.shape, (2, 5, 16))


class TestGPTLanguageModel(unittest.TestCase):
    """Test GPT language model"""
    
    def test_model_creation(self):
        """Test model creation with different configurations"""
        model = GPTLanguageModel(vocab_size=100, n_embd=64, n_head=4, n_layer=2)
        
        # Check model parameters
        self.assertEqual(model.vocab_size, 100)
        self.assertEqual(model.n_embd, 64)
        self.assertEqual(model.n_head, 4)
        self.assertEqual(model.n_layer, 2)
        
        # Check parameter count
        params = list(model.parameters())
        self.assertGreater(len(params), 0)
        
    def test_forward_pass(self):
        """Test model forward pass"""
        model = GPTLanguageModel(vocab_size=50, n_embd=32, n_head=4, n_layer=2, block_size=10)
        idx = Tensor(np.random.randint(0, 50, (2, 5)))
        
        logits, loss = model(idx)
        
        # Check output shape
        self.assertEqual(logits.data.shape, (2, 5, 50))
        self.assertIsNone(loss)
        
    def test_forward_with_targets(self):
        """Test model forward pass with targets"""
        model = GPTLanguageModel(vocab_size=50, n_embd=32, n_head=4, n_layer=2, block_size=10)
        idx = Tensor(np.random.randint(0, 50, (2, 5)))
        targets = Tensor(np.random.randint(0, 50, (2, 5)))
        
        logits, loss = model(idx, targets)
        
        # Check output shape
        self.assertEqual(logits.data.shape, (2, 5, 50))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.data.item(), 0)
        
    def test_generation(self):
        """Test text generation"""
        model = GPTLanguageModel(vocab_size=50, n_embd=32, n_head=4, n_layer=2, block_size=10)
        idx = Tensor(np.random.randint(0, 50, (1, 3)))
        
        generated = model.generate(idx, max_new_tokens=5)
        
        # Check output shape
        self.assertEqual(generated.shape, (1, 8))  # 3 + 5 tokens
        
    def test_sequence_cropping(self):
        """Test sequence cropping utility"""
        model = GPTLanguageModel(vocab_size=50, n_embd=32, n_head=4, n_layer=2, block_size=5)
        long_sequence = Tensor(np.random.randint(0, 50, (1, 10)))
        
        cropped = model.crop_sequence(long_sequence, 5)
        self.assertEqual(cropped.data.shape, (1, 5))
        
        # Should be the last 5 tokens
        np.testing.assert_array_equal(cropped.data, long_sequence.data[:, -5:])


class TestAdam(unittest.TestCase):
    """Test Adam optimizer"""
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        model = Linear(2, 3)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        self.assertEqual(optimizer.lr, 0.01)
        self.assertEqual(len(optimizer.param_groups[0]['params']), 2)  # weight + bias
        
    def test_optimization_step(self):
        """Test optimization step"""
        model = Linear(2, 1)
        optimizer = Adam(model.parameters(), lr=0.1)
        
        # Store initial parameters
        initial_weight = model.weight.data.copy()
        initial_bias = model.bias.data.copy()
        
        # Forward pass
        x = Tensor([[1, 2]], requires_grad=True)
        y = model(x)
        loss = y.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        # Parameters should have changed
        self.assertFalse(np.array_equal(model.weight.data, initial_weight))
        self.assertFalse(np.array_equal(model.bias.data, initial_bias))
        
    def test_zero_grad(self):
        """Test zero_grad functionality"""
        model = Linear(2, 1)
        optimizer = Adam(model.parameters())
        
        # Set some gradients
        for param in model.parameters():
            param.grad = np.ones_like(param.data)
            
        optimizer.zero_grad()
        
        # Check all gradients are zero
        for param in model.parameters():
            np.testing.assert_array_equal(param.grad, np.zeros_like(param.data))


class TestModelState(unittest.TestCase):
    """Test model saving and loading"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.json")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        os.rmdir(self.temp_dir)
        
    def test_save_and_load_model(self):
        """Test saving and loading a model"""
        # Create and train a simple model
        original_model = GPTLanguageModel(vocab_size=50, n_embd=32, n_head=4, n_layer=2)
        
        # Save model
        additional_info = {"epoch": 10, "loss": 0.5}
        save_model(original_model, self.model_path, additional_info)
        
        # Check file exists
        self.assertTrue(os.path.exists(self.model_path))
        
        # Load model
        loaded_model, loaded_info = load_model(self.model_path)
        
        # Check configuration matches
        self.assertEqual(loaded_model.vocab_size, original_model.vocab_size)
        self.assertEqual(loaded_model.n_embd, original_model.n_embd)
        self.assertEqual(loaded_model.n_head, original_model.n_head)
        self.assertEqual(loaded_model.n_layer, original_model.n_layer)
        
        # Check additional info
        self.assertEqual(loaded_info["epoch"], 10)
        self.assertEqual(loaded_info["loss"], 0.5)
        
        # Check parameters are the same
        orig_params = list(original_model.parameters())
        loaded_params = list(loaded_model.parameters())
        
        for orig_param, loaded_param in zip(orig_params, loaded_params):
            np.testing.assert_array_almost_equal(orig_param.data, loaded_param.data)
            
    def test_model_exists(self):
        """Test model existence check"""
        self.assertFalse(model_exists(self.model_path))
        
        # Create a model and save it
        model = GPTLanguageModel(vocab_size=10, n_embd=16, n_head=2, n_layer=1)
        save_model(model, self.model_path)
        
        self.assertTrue(model_exists(self.model_path))
        
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model"""
        with self.assertRaises(FileNotFoundError):
            load_model("nonexistent_model.json")


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire framework"""
    
    def test_simple_training_loop(self):
        """Test a complete training loop"""
        # Create a simple model
        model = GPTLanguageModel(vocab_size=10, n_embd=16, n_head=2, n_layer=1, block_size=5)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        # Create dummy data
        batch_size = 2
        seq_length = 4
        inputs = Tensor(np.random.randint(0, 10, (batch_size, seq_length)))
        targets = Tensor(np.random.randint(0, 10, (batch_size, seq_length)))
        
        # Training step
        initial_loss = None
        for step in range(3):
            # Forward pass
            logits, loss = model(inputs, targets)
            
            if initial_loss is None:
                initial_loss = loss.data.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check loss is a number
            self.assertIsInstance(loss.data.item(), float)
            
        # Loss should exist and be positive
        self.assertGreater(initial_loss, 0)
        
    def test_model_inference(self):
        """Test model inference without gradients"""
        model = GPTLanguageModel(vocab_size=20, n_embd=32, n_head=4, n_layer=2, block_size=10)
        
        # Set model to inference mode (disable dropout)
        for module in [model] + list(model._modules.values()):
            if hasattr(module, 'training'):
                module.training = False
        
        # Generate text
        start_tokens = Tensor(np.random.randint(0, 20, (1, 3)))
        generated = model.generate(start_tokens, max_new_tokens=5)
        
        # Check output
        self.assertEqual(generated.shape, (1, 8))
        self.assertTrue(np.all(generated >= 0))
        self.assertTrue(np.all(generated < 20))
        
