using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{

    /// <summary>
    /// Used in the Brain to normalize neuron values and calculate its derivatives.
    /// </summary>
    public interface INeuronActivation
    {
        /// <summary>
        /// Normalizes given neuron matrix values in a class specific way.
        /// </summary>
        /// <param name="neurons">The neuron vector to be normalized.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        Matrix<double> Activate(Matrix<double> neurons, int layer);

        /// <summary>
        /// Returns a derivative of a class specific funtion for all values in a given neuron matrix.
        /// </summary>
        /// <param name="neurons">The neuron vector for the function to be derivatived at.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        Matrix<double> Derive(Matrix<double> neurons, int layer);
    }

    /// <summary>
    /// Is synonymous with no activation function. Derivative is always 1.
    /// </summary>
    public class RawActivation : INeuronActivation
    {
        /// <summary>
        /// Returns raw neurons.
        /// </summary>
        /// <param name="neurons">The neuron vector to be normalized.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Activate(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => x);
        }

        /// <summary>
        /// Returns a column matrix of ones.
        /// </summary>
        /// <param name="neurons">The neuron vector for the function to be derivatived at.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Derive(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => 1.0);
        }
    }

    /// <summary>
    /// The logistic(sigmoid) activation. Squeezes values from -∞ to ∞ into (0; 1) range.
    /// </summary>
    public class LogisticActivation : INeuronActivation
    {
        //https://calculus.subwiki.org/wiki/Logistic_function
        /// <summary>
        /// Normalizes given neuron matrix by applying logistic function to each value.
        /// </summary>
        /// <param name="neurons">The neuron vector to be normalized.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Derive(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => { x = SpecialFunctions.Logistic(x); return x * (1 - x); });
        }

        /// <summary>
        /// Returns column matrix of logistic function derivative applied to all values in a given neuron matrix.
        /// </summary>
        /// <param name="neurons">The neuron vector for the function to be derivatived at.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Activate(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => SpecialFunctions.Logistic(x));
        }
    }

    /// <summary>
    /// The Hyperbolic Secant function(Sech) activation. Its a bell shaped function where f(0)=1 and infinities converge to 0.
    /// </summary>
    public class SechActivation : INeuronActivation
    {
        //https://en.wikipedia.org/wiki/Hyperbolic_function
        /// <summary>
        /// Normalizes given neuron matrix by applying hyperbolic function to each value.
        /// </summary>
        /// <param name="neurons">The neuron vector to be normalized.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Derive(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => -Trig.Sech(x) * Trig.Tanh(x));

        }

        /// <summary>
        /// Returns column matrix of hyperbolic function derivative applied to all values in a given neuron matrix.
        /// </summary>
        /// <param name="neurons">The neuron vector for the function to be derivatived at.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Activate(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => Trig.Sech(x));
        }
    }

    /// <summary>
    /// The Hyperbolic Tangent function(Tanh) activation.
    /// </summary>
    public class TanhActivation : INeuronActivation
    {
        //https://en.wikipedia.org/wiki/Hyperbolic_function
        /// <summary>
        /// Normalizes given neuron matrix by applying hyperbolic tangens function to each value.
        /// </summary>
        /// <param name="neurons">The neuron vector to be normalized.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Derive(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => { x = Trig.Sech(x); return x * x; });

        }

        /// <summary>
        /// Returns column matrix of hyperbolic tangens function derivative applied to all values in a given neuron matrix.
        /// </summary>
        /// <param name="neurons">The neuron vector for the function to be derivatived at.</param>
        /// <param name="layer">The neuron layer number, with input neurons being layer zero.</param>
        public Matrix<double> Activate(Matrix<double> neurons, int layer)
        {
            return neurons.Map((x) => Trig.Tanh(x));
        }
    }



}
