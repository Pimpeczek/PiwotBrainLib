using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{
    class Brain
    {
        protected readonly MathNet.Numerics.Distributions.IContinuousDistribution distribution = new MathNet.Numerics.Distributions.Normal();
        INeuronActivation neuronActivation = new LogisticActivation();

        public INeuronActivation NeuronActivation
        {
            get
            {
                return neuronActivation;
            }
            set
            {
                neuronActivation = value ?? throw new ArgumentNullException("neuronActivation");
            }
        }

        protected int[] layerCounts;
        protected int neuronLayerCount;
        protected int synapsLayerCount;

        public int InputNeuronCount { get; protected set; }
        public int OutputNeuronCount { get; protected set; }

        public int TotalNeuronLayers { get; protected set; }
        public int TotalSynapsLayers { get; protected set; }

        protected Matrix<double>[] activeNeurons;
        protected Matrix<double>[] derivedNeurons;
        protected Matrix<double>[] rawNeurons;
        protected Matrix<double>[] biases;
        protected Matrix<double>[] synapses;


        public Brain(int inputNeurons, int hiddenNeurons, int outputNeurons)
        {

            SetupLayerCounts(inputNeurons, new int[]{hiddenNeurons}, outputNeurons);
            BuildBrain();
        }
        public Brain(int inputNeurons, int[] hiddenNeurons, int outputNeurons)
        {

            SetupLayerCounts(inputNeurons, hiddenNeurons, outputNeurons);
            BuildBrain();
        }

        protected void SetupLayerCounts(int inputNeurons, int[] hiddenNeurons, int outputNeurons)
        {
            if (hiddenNeurons == null)
            {
                throw new ArgumentNullException("hiddenNeurons");
            }

            if (inputNeurons < 1)
            {
                throw new ArgumentOutOfRangeException("inputNeurons");
            }

            if (outputNeurons < 1)
            {
                throw new ArgumentOutOfRangeException("outputNeurons");
            }
            InputNeuronCount = inputNeurons;
            OutputNeuronCount = outputNeurons;
            neuronLayerCount = hiddenNeurons.Length + 2;
            TotalNeuronLayers = neuronLayerCount;
            synapsLayerCount = neuronLayerCount - 1;
            TotalSynapsLayers = synapsLayerCount;
            layerCounts = new int[neuronLayerCount];
            layerCounts[0] = inputNeurons;
            layerCounts[synapsLayerCount] = outputNeurons;
            for (int i = 1; i < synapsLayerCount; i++)
            {
                if(hiddenNeurons[i-1] < 1)
                {
                    throw new ArgumentOutOfRangeException($"hiddenNeurons[{i - 1}]");
                }
                layerCounts[i] = hiddenNeurons[i - 1];
            }
        }

        protected void BuildBrain()
        {
            InstantiateFrame();
            PopulateFrame();
        }

        protected void InstantiateFrame()
        {
            neuronLayerCount = layerCounts.Length;
            synapsLayerCount = neuronLayerCount - 1;
            activeNeurons = new Matrix<double>[neuronLayerCount];
            derivedNeurons = new Matrix<double>[neuronLayerCount];
            rawNeurons = new Matrix<double>[neuronLayerCount];
            biases = new Matrix<double>[synapsLayerCount];
            synapses = new Matrix<double>[synapsLayerCount];
        }

        protected void PopulateFrame()
        {
            for (int i = 0; i < synapsLayerCount; i++)
            {
                biases[i] = Matrix<double>.Build.Random(layerCounts[i + 1], 1, distribution);

                synapses[i] = Matrix<double>.Build.Random(layerCounts[i + 1], layerCounts[i], distribution);
            }
        }

        public Vector<double> Calculate(Vector<double> input)
        {
            activeNeurons[0] = input.ToColumnMatrix();
            rawNeurons[0] = input.ToColumnMatrix();
            for (int i = 1; i < neuronLayerCount; i++)
            {
                rawNeurons[i] = synapses[i - 1] * activeNeurons[i - 1] + biases[i - 1];
                activeNeurons[i] = neuronActivation.Activate(rawNeurons[i]);
            }
            return activeNeurons[synapsLayerCount].Column(0);
        }

        public void ApplyGradients(Matrix<double>[] synapsGradient, Matrix<double>[] biasGradient)
        {
            for (int i = synapsLayerCount - 1; i >= 0; i--)
            {
                synapses[i] -= synapsGradient[i];
                biases[i] -= biasGradient[i];
            }
        }

        public (Matrix<double>[], Matrix<double>[], double) CalculateOneGradients(Vector<double> input, Vector<double> output)
        {
            if (input == null)
            {
                throw new ArgumentNullException("input");
            }

            if (output == null)
            {
                throw new ArgumentNullException("output");
            }
            return CalculateOneGradients(input.ToColumnMatrix(), output.ToColumnMatrix());
        }

        public (Matrix<double>[], Matrix<double>[], double) CalculateOneGradients(Matrix<double> input, Matrix<double> output)
        {
            if(input == null)
            {
                throw new ArgumentNullException("input");
            }

            if(output == null)
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
                derivedNeurons[i] = neuronActivation.Derive(rawNeurons[i]);
                activeNeurons[i] = neuronActivation.Activate(rawNeurons[i]);
            }
            Matrix<double>[] synapsDerivatives = new Matrix<double>[synapsLayerCount];
            Matrix<double>[] biasDerivatives = new Matrix<double>[synapsLayerCount];
            Matrix<double> costDerivatives = (activeNeurons[synapsLayerCount] - output) * 2;
            Matrix<double> neuronTailProduct;
            Matrix<double> onesRow;
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

        public Matrix<double>[] GetSynapsGradientFrame()
        {
            Matrix<double>[] frame = new Matrix<double>[synapsLayerCount];
            for (int i = 0; i < synapsLayerCount; i++)
            {
                frame[i] = Matrix<double>.Build.Dense(layerCounts[i + 1], layerCounts[i]);
            }
            return frame;
        }

        public Matrix<double>[] GetBiasGradientFrame()
        {
            Matrix<double>[] frame = new Matrix<double>[synapsLayerCount];
            for (int i = 0; i < synapsLayerCount; i++)
            {
                frame[i] = Matrix<double>.Build.Dense(layerCounts[i + 1], 1);
            }
            return frame;
        }

    }
}
