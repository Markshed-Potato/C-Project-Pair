namespace MlpChurn.Network;

/// <summary>Fully connected layer of neurons.</summary>
public sealed class Layer
{
    public Neuron[] Neurons { get; }
    public ActivationType Activation { get; }
    public double[] LastInputs { get; private set; } = Array.Empty<double>();

    public Layer(int inputCount, int neuronCount, ActivationType activation, Random rng)
    {
        Activation = activation;
        Neurons = Enumerable.Range(0, neuronCount).Select(_ => new Neuron(inputCount, rng)).ToArray();
    }

    public double[] Forward(double[] inputs)
    {
        LastInputs = inputs;
        var outputs = new double[Neurons.Length];
        for (int i = 0; i < Neurons.Length; i++)
            outputs[i] = Neurons[i].Forward(inputs, Activation);
        return outputs;
    }
}
