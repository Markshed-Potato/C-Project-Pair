namespace MlpChurn.Network;

/// <summary>
/// 3-layer MLP: Input -> Hidden -> Output (single sigmoid neuron for binary classification).
/// Trained with per-sample SGD backpropagation on MSE loss: L = 0.5 * (o - t)^2.
/// </summary>
public sealed class NeuralNetwork
{
    private readonly Layer _hidden;
    private readonly Layer _output;

    public NeuralNetwork(int inputSize, int hiddenSize, ActivationType hiddenActivation = ActivationType.Sigmoid, int seed = 1)
    {
        var rng = new Random(seed);
        _hidden = new Layer(inputSize, hiddenSize, hiddenActivation, rng);
        _output = new Layer(hiddenSize, 1, ActivationType.Sigmoid, rng);
    }

    /// <summary>Forward propagation; returns probability of class 1.</summary>
    public double Predict(double[] input) => _output.Forward(_hidden.Forward(input))[0];

    public double[] PredictBatch(double[][] inputs) => inputs.Select(Predict).ToArray();

    /// <summary>One SGD step on one sample. Returns the squared error before the update.</summary>
    public double TrainSample(double[] input, double target, double learningRate)
    {
        double prediction = Predict(input); // forward pass (caches outputs)
        double error = prediction - target;

        // 1) Output delta: dL/dz = (o - t) * f'(o)
        var outNeuron = _output.Neurons[0];
        outNeuron.Delta = error * Activation.DerivativeFromOutput(_output.Activation, outNeuron.Output);

        // 2) Hidden deltas (use OLD output weights, so compute before updating them):
        //    delta_j = f'(h_j) * delta_out * w_out_j
        for (int j = 0; j < _hidden.Neurons.Length; j++)
        {
            var h = _hidden.Neurons[j];
            h.Delta = Activation.DerivativeFromOutput(_hidden.Activation, h.Output)
                      * outNeuron.Delta * outNeuron.Weights[j];
        }

        // 3) Update weights and biases: w -= lr * delta * input
        UpdateLayer(_output, learningRate);
        UpdateLayer(_hidden, learningRate);

        return error * error;
    }

    private static void UpdateLayer(Layer layer, double lr)
    {
        foreach (var n in layer.Neurons)
        {
            for (int i = 0; i < n.Weights.Length; i++)
                n.Weights[i] -= lr * n.Delta * layer.LastInputs[i];
            n.Bias -= lr * n.Delta;
        }
    }
}
