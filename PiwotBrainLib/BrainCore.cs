using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{
    class BrainCore
    {
        protected readonly MathNet.Numerics.Distributions.IContinuousDistribution distribution = new MathNet.Numerics.Distributions.Normal();
        protected INeuronActivation neuronActivation = new LogisticActivation();

        /// <summary>
        /// A class used to activate each neuron.
        /// </summary>
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
        protected Matrix<double>[] biases;
        protected Matrix<double>[] synapses;

        /// <param name="inputNeurons">Number of input parameters.</param>
        /// <param name="hiddenNeurons">Number of neurons on the hidden layer.</param>
        /// <param name="outputNeurons">Number of output neurons.</param>
        public BrainCore(int inputNeurons, int hiddenNeurons, int outputNeurons)
        {

            SetupLayerCounts(inputNeurons, new int[] { hiddenNeurons }, outputNeurons);
            BuildBrain();
        }

        /// <param name="inputNeurons">Number of input parameters.</param>
        /// <param name="hiddenNeurons">Array containing number of neurons on each hidden layer.</param>
        /// <param name="outputNeurons">Number of output neurons.</param>
        public BrainCore(int inputNeurons, int[] hiddenNeurons, int outputNeurons)
        {

            SetupLayerCounts(inputNeurons, hiddenNeurons, outputNeurons);
            BuildBrain();
        }
        /// <summary>
        /// Uses existing brain to create new one.
        /// </summary>
        /// <param name="baseBrain">Brain to be copied.</param>
        public BrainCore(BrainCore baseBrain)
        {
            biases = baseBrain.biases;
            synapses = baseBrain.synapses;
            layerCounts = baseBrain.layerCounts;
            int[] hiddenNeurons = new int[layerCounts.Length - 2];
            for (int i = 0; i < hiddenNeurons.Length;)
            {
                hiddenNeurons[i] = layerCounts[++i];
            }
            SetupLayerCounts(layerCounts[0], hiddenNeurons, layerCounts[layerCounts.Length - 1]);
            activeNeurons = new Matrix<double>[neuronLayerCount];
            neuronActivation = baseBrain.neuronActivation;
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
                if (hiddenNeurons[i - 1] < 1)
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
            activeNeurons = new Matrix<double>[neuronLayerCount];
            
            biases = new Matrix<double>[synapsLayerCount];
            synapses = new Matrix<double>[synapsLayerCount];
        }

        /// <summary>
        /// Assigns a random number to each field in synapses and biases arrays.
        /// </summary>
        protected void PopulateFrame()
        {
            for (int i = 0; i < synapsLayerCount; i++)
            {
                biases[i] = Matrix<double>.Build.Random(layerCounts[i + 1], 1, distribution);

                synapses[i] = Matrix<double>.Build.Random(layerCounts[i + 1], layerCounts[i], distribution);
            }
        }

        /// <summary>
        /// Performs learned function on a given input and returns the result.
        /// </summary>
        /// <param name="input">Data to be interpreted</param>
        public Vector<double> Calculate(Vector<double> input)
        {
            activeNeurons[0] = input.ToColumnMatrix();
            for (int i = 1; i < neuronLayerCount; i++)
            {
                activeNeurons[i] = synapses[i - 1] * activeNeurons[i - 1];
                activeNeurons[i] += biases[i - 1];
                activeNeurons[i] = neuronActivation.Activate(activeNeurons[i], i);
            }
            return activeNeurons[synapsLayerCount].Column(0);
        }


        /// <summary>
        /// Returns empty synaps array.
        /// </summary>
        protected Matrix<double>[] GetSynapsGradientFrame()
        {
            Matrix<double>[] frame = new Matrix<double>[synapsLayerCount];
            for (int i = 0; i < synapsLayerCount; i++)
            {
                frame[i] = Matrix<double>.Build.Dense(layerCounts[i + 1], layerCounts[i]);
            }
            return frame;
        }


        /// <summary>
        /// Returns empty bias array.
        /// </summary>
        protected Matrix<double>[] GetBiasGradientFrame()
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
