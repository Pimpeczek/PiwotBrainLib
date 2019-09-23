using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{
    class OpenBrain : BrainCore
    {

        protected Matrix<double>[] derivedNeurons;
        protected Matrix<double>[] rawNeurons;
        Matrix<double>[] synapsDerivatives;
        Matrix<double>[] biasDerivatives;
        Matrix<double> costDerivatives;
        Matrix<double> neuronTailProduct;
        Matrix<double> onesRow;

        public OpenBrain(int inputNeurons, int hiddenNeurons, int outputNeurons) : base(inputNeurons, hiddenNeurons, outputNeurons)
        {
            BuildOpenBrain();
        }

        public OpenBrain(int inputNeurons, int[] hiddenNeurons, int outputNeurons) : base(inputNeurons, hiddenNeurons, outputNeurons)
        {
            BuildOpenBrain();
        }

        /// <summary>
        /// Uses already existing brain to 
        /// </summary>
        /// <param name="brainCore"></param>
        public OpenBrain(BrainCore brainCore) : base(brainCore)
        {
            BuildOpenBrain();
        }

        protected void BuildOpenBrain()
        {
            derivedNeurons = new Matrix<double>[neuronLayerCount];
            rawNeurons = new Matrix<double>[neuronLayerCount];
            synapsDerivatives = new Matrix<double>[synapsLayerCount];
            biasDerivatives = new Matrix<double>[synapsLayerCount];
        }

        public void ApplyGradients(Matrix<double>[] synapsGradient, Matrix<double>[] biasGradient)
        {
            for (int i = synapsLayerCount - 1; i >= 0; i--)
            {
                synapses[i] -= synapsGradient[i];
                biases[i] -= biasGradient[i];
            }
        }

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

            activeNeurons[0] = input;
            rawNeurons[0] = input;
            double error;
            for (int i = 1; i < neuronLayerCount; i++)
            {
                rawNeurons[i] = synapses[i - 1] * activeNeurons[i - 1] + biases[i - 1];
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
            biasDerivatives[synapsLayerCount - 1] = costDerivatives.PointwiseMultiply(derivedNeurons[synapsLayerCount]);
            for (int layer = synapsLayerCount - 2; layer >= 0; layer--)
            {

                onesRow = Matrix<double>.Build.Dense(1, layerCounts[layer], 1);
                neuronTailProduct = derivedNeurons[layer + 1] * activeNeurons[layer].Transpose();
                biasDerivatives[layer] = synapsDerivatives[layer].PointwiseMultiply(derivedNeurons[layer + 1]);
                synapsDerivatives[layer] = synapsDerivatives[layer] * onesRow;
                synapsDerivatives[layer] = synapsDerivatives[layer].PointwiseMultiply(neuronTailProduct);

            }
            return (synapsDerivatives, biasDerivatives, error);
        }

    }
}
