using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace PiwotBrainLib
{
    class InputNormalization
    {
        /// <summary>
        /// Returns copy of input vector. Mostly useless.
        /// </summary>
        /// <param name="values">Value vector to be normalized.</param>
        public static Vector<double> RawValues(Vector<double> values)
        {
            return values.Map((x) => x);
        }

        /// <summary>
        /// Applies logistic function(sigmoid) to every value of a vector.
        /// </summary>
        /// <param name="values">Value vector to be normalized.</param>
        public static Vector<double> Logistic(Vector<double> values)
        {
            return values.Map((x) => SpecialFunctions.Logistic(x));
        }

        /// <summary>
        /// Applies Hyperbolic Secant function(Sech) to every value of a vector. Its a bell shaped function where f(0)=1 and infinities converge to 0.
        /// </summary>
        /// <param name="values">Value vector to be normalized.</param>
        public static Vector<double> Sech(Vector<double> values)
        {
            return values.Map((x) => SpecialFunctions.Logistic(x));
        }


    }
}
