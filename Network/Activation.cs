namespace MlpChurn.Network;

public enum ActivationType { Sigmoid, ReLU }

public static class Activation
{
    public static double Apply(ActivationType t, double x) => t switch
    {
        ActivationType.Sigmoid => 1.0 / (1.0 + Math.Exp(-Math.Clamp(x, -500, 500))),
        ActivationType.ReLU => Math.Max(0.0, x),
        _ => throw new ArgumentOutOfRangeException(nameof(t))
    };

    /// <summary>Derivative expressed in terms of the neuron OUTPUT (cheaper, no recompute).</summary>
    public static double DerivativeFromOutput(ActivationType t, double output) => t switch
    {
        ActivationType.Sigmoid => output * (1.0 - output),
        ActivationType.ReLU => output > 0 ? 1.0 : 0.0,
        _ => throw new ArgumentOutOfRangeException(nameof(t))
    };
}
