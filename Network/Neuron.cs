namespace MlpChurn.Network;

/// <summary>Single neuron: weights, bias, cached output and backprop delta.</summary>
public sealed class Neuron
{
    public double[] Weights { get; }
    public double Bias { get; set; }
    public double Output { get; set; }
    public double Delta { get; set; }

    public Neuron(int inputCount, Random rng)
    {
        // Xavier-style init keeps early activations in a sane range.
        double limit = Math.Sqrt(6.0 / inputCount);
        Weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++)
            Weights[i] = (rng.NextDouble() * 2 - 1) * limit;
        Bias = 0.0;
    }

    public double Forward(double[] inputs, ActivationType activation)
    {
        double sum = Bias;
        for (int i = 0; i < Weights.Length; i++)
            sum += Weights[i] * inputs[i]; // dot product
        Output = Activation.Apply(activation, sum);
        return Output;
    }
}
