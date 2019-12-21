using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{
    public class OpenBrain : BrainCore
    {
        #region Variables
        protected Matrix<double>[] derivedNeurons;
        protected Matrix<double>[] rawNeurons;
        Matrix<double>[] synapsDerivatives;
        Matrix<double>[] biasDerivatives;
        Matrix<double> costDerivatives;
        Matrix<double> neuronTailProduct;
        Matrix<double> onesRow;
        #endregion

        #region Constructors
        /// <param name="inputNeurons">Number of input parameters.</param>
        /// <param name="hiddenNeurons">Number of neurons on the hidden layer.</param>
        /// <param name="outputNeurons">Number of output neurons.</param>
        public OpenBrain(int inputNeurons, int hiddenNeurons, int outputNeurons) : base(inputNeurons, hiddenNeurons, outputNeurons)
        {
            BuildOpenBrain();
        }

        /// <param name="inputNeurons">Number of input parameters.</param>
        /// <param name="hiddenNeurons">Array containing number of neurons on each hidden layer.</param>
        /// <param name="outputNeurons">Number of output neurons.</param>
        public OpenBrain(int inputNeurons, int[] hiddenNeurons, int outputNeurons) : base(inputNeurons, hiddenNeurons, outputNeurons)
        {
            BuildOpenBrain();
        }

        /// <summary>
        /// Uses already existing brain to build this instance of OpenBrain.
        /// </summary>
        /// <param name="brainCore">Brain to be used as a base.</param>
        public OpenBrain(BrainCore brainCore) : base(brainCore)
        {
            BuildOpenBrain();
        }
        #endregion
        #region Setup
        protected void BuildOpenBrain()
        {
            derivedNeurons = new Matrix<double>[neuronLayerCount];
            rawNeurons = new Matrix<double>[neuronLayerCount];
            synapsDerivatives = new Matrix<double>[synapsLayerCount];
            biasDerivatives = new Matrix<double>[neuronLayerCount];
        }
        #endregion
        #region Calculations
        /// <summary>
        /// 
        /// </summary>
        /// <param name="synapsGradient">Array of synaps gradient matrices, starting from the deepest layer.</param>
        /// <param name="biasGradient">Array of bias gradient matrices, starting from the deepest layer.</param>
        public void ApplyGradients(Matrix<double>[] synapsGradient, Matrix<double>[] biasGradient)
        {
            biases[synapsLayerCount] -= biasGradient[synapsLayerCount];

            for (int i = synapsLayerCount - 1; i >= 0; i--)
            {
                synapses[i] -= synapsGradient[i];
                biases[i] -= biasGradient[i];
            }
        }

        /// <summary>
        /// Returns gradients of all synapses and biases as an array of matrices.
        /// </summary>
        /// <param name="input">The learning data formated to a vector.</param>
        /// <param name="output">The expected output data formated to a vector.</param>
        public (Matrix<double>[], Matrix<double>[], double) CalculateGradients(Vector<double> input, Vector<double> output)
        {
            if (input == null)
            {
                throw new ArgumentNullException("input");
            }

            if (output == null)
            {
                throw new ArgumentNullException("output");
            }
            return CalculateGradients(input.ToColumnMatrix(), output.ToColumnMatrix());
        }


        /// <summary>
        /// Returns gradients of all synapses and biases and the MeanSquaredError a touple of as two arrays of matrices and a double.
        /// </summary>
        /// <param name="input">The learning data formated to column matrix.</param>
        /// <param name="output">The expected output data formated to column matrix.</param>
        public (Matrix<double>[], Matrix<double>[], double) CalculateGradients(Matrix<double> input, Matrix<double> output)
        {
            if (input == null)
            {
                throw new ArgumentNullException("input");
            }

            if (output == null)
            {
                throw new ArgumentNullException("output");
            }

            if (input.RowCount != InputNeuronCount || input.ColumnCount > 1)
            {
                throw new ArgumentException("input");
            }

            if (output.RowCount != OutputNeuronCount || output.ColumnCount > 1)
            {
                throw new ArgumentException("output");
            }

            
            rawNeurons[0] = input + biases[0];
            derivedNeurons[0] = neuronActivation.Derive(rawNeurons[0], 0);
            activeNeurons[0] = neuronActivation.Activate(rawNeurons[0], 0);
            double error;

            for (int i = 1; i < neuronLayerCount; i++)
            {
                rawNeurons[i] = synapses[i - 1] * activeNeurons[i - 1] + biases[i];
                derivedNeurons[i] = neuronActivation.Derive(rawNeurons[i], i);
                activeNeurons[i] = neuronActivation.Activate(rawNeurons[i], i);
            }

            costDerivatives = (activeNeurons[synapsLayerCount] - output) * 2;
            error = (activeNeurons[synapsLayerCount] - output).Map((x) => x * x).ColumnSums()[0];
            synapsDerivatives[synapsLayerCount - 1] = costDerivatives;

            for (int layer = synapsLayerCount - 2; layer >= 0; layer--)
            {
                synapsDerivatives[layer] = synapsDerivatives[layer + 1].PointwiseMultiply(derivedNeurons[layer + 2]);
                synapsDerivatives[layer] = synapses[layer + 1].Transpose() * synapsDerivatives[layer];
            }
            synapsDerivatives[synapsLayerCount - 1] = costDerivatives.PointwiseMultiply(derivedNeurons[synapsLayerCount]) * activeNeurons[synapsLayerCount - 1].Transpose();
            biasDerivatives[synapsLayerCount] = costDerivatives.PointwiseMultiply(derivedNeurons[synapsLayerCount]);


            for (int layer = synapsLayerCount - 2; layer >= 0; layer--)
            {

                onesRow = Matrix<double>.Build.Dense(1, layerCounts[layer], 1);
                neuronTailProduct = derivedNeurons[layer + 1] * activeNeurons[layer].Transpose();
                biasDerivatives[layer + 1] = synapsDerivatives[layer].PointwiseMultiply(derivedNeurons[layer + 1]);
                synapsDerivatives[layer] = synapsDerivatives[layer] * onesRow;
                synapsDerivatives[layer] = synapsDerivatives[layer].PointwiseMultiply(neuronTailProduct);

            }

            biasDerivatives[0] = (synapses[0].Transpose() * biasDerivatives[1]).PointwiseMultiply(derivedNeurons[0]);

            return (synapsDerivatives, biasDerivatives, error);
        }
        #endregion
    }
}
