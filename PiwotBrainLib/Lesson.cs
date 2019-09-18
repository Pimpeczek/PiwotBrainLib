using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{
    class Example
    {
        public Matrix<double> input;
        public Matrix<double> output;

        public Example(Matrix<double> input, Matrix<double> output)
        {
            this.input = input;
            this.output = output;
        }
        public Example()
        {

        }
    }
}
