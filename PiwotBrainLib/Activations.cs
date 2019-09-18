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
    interface INeuronActivation
    {
        /// <summary>
        /// Normalizes a given value in a class specific way.
        /// </summary>
        /// <param name="neurons">The neuron vector to be normalized.</param>
        /// <returns>xDS</returns>
        Matrix<double> Activate(Matrix<double> neurons);

        /// <summary>
        /// Returns a derivative of a class specific funtion at a given value.
        /// </summary>
        /// <param name="neurons">The neuron vector for the function to be derivatived at.</param>
        /// <returns>xDS</returns>
        Matrix<double> Derive(Matrix<double> neurons);
    }

    /// <summary>
    /// Is synonymous with no activation function. Derivative is always 1.
    /// </summary>
    class RawActivation : INeuronActivation
    {

        public Matrix<double> Activate(Matrix<double> neurons)
        {
            return neurons.Map((x) => x);
        }

        public Matrix<double> Derive(Matrix<double> neurons)
        {
            return neurons.Map((x) => 1.0);
        }
    }

    /// <summary>
    /// The logistic(sigmoid) activation. Squeezes values from -∞ to ∞ into (0; 1) range.
    /// </summary>
    class LogisticActivation : INeuronActivation
    {
        //https://calculus.subwiki.org/wiki/Logistic_function
        public Matrix<double> Derive(Matrix<double> neurons)
        {


            return neurons.Map((x) => { x = SpecialFunctions.Logistic(x); return x * (1 - x); });

        }
        public Matrix<double> Activate(Matrix<double> neurons)
        {
            return neurons.Map((x) => SpecialFunctions.Logistic(x));
        }
    }

    /// <summary>
    /// The Hyperbolic Secant function(Sech) activation. Its a bell shaped function where f(0)=1 and infinities converge to 0.
    /// </summary>
    class SechActivation : INeuronActivation
    {
        //https://en.wikipedia.org/wiki/Hyperbolic_function
        public Matrix<double> Derive(Matrix<double> neurons)
        {
            return neurons.Map((x) => -Trig.Sech(x) * Trig.Tanh(x));

        }
        public Matrix<double> Activate(Matrix<double> neurons)
        {
            return neurons.Map((x) => Trig.Sech(x));
        }
    }

    /// <summary>
    /// The Hyperbolic Tangent function(Tanh) activation.
    /// </summary>
    class TanhActivation : INeuronActivation
    {
        //https://en.wikipedia.org/wiki/Hyperbolic_function
        public Matrix<double> Derive(Matrix<double> neurons)
        {
            return neurons.Map((x) => { x = Trig.Sech(x); return x * x; });

        }
        public Matrix<double> Activate(Matrix<double> neurons)
        {
            return neurons.Map((x) => Trig.Tanh(x));
        }
    }



}
