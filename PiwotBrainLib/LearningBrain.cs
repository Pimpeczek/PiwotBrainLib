using System;
using System.Collections;
using System.Linq;
using System.Text;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{
    class LearningBrain : OpenBrain
    {
       
        protected Matrix<double>[] synapsGradient;
        protected Matrix<double>[] biasGradient;
        protected (Matrix<double>[], Matrix<double>[], double) gradientTouple;
        protected Matrix<double>[] synapsGradientMomentum;
        protected Matrix<double>[] biasGradientMomentum;
        protected Matrix<double>[,] learningData;

        /// <summary>
        /// The number of blocks learned during one learning session.
        /// </summary>
        public int BlocksDone { get; protected set; } = 0;
        /// <summary>
        /// The number of examples learned during one learning session.
        /// </summary>
        public long ExamplesDone { get; protected set; } = 0;

        /// <summary>
        /// The total number of blocks learned during all learning sessions.
        /// </summary>
        public int TotalBlocksDone { get; protected set; } = 0;
        /// <summary>
        /// The total number of examples learned during all learning sessions.
        /// </summary>
        public long TotalExamplesDone { get; protected set; } = 0;

        protected Vector<double> errors;
        int lastErrorPosition = 0;

        /// <summary>
        /// The average calculated MeanSquaredErrors relative to expected output.
        /// </summary>
        public double MeanSquaredError { get; protected set; } = double.PositiveInfinity;
        protected int errorMemoryLenght = 10;

        /// <summary>
        /// Number of saved MeanSquaredErrors used to calculate the average.
        /// </summary>
        public int ErrorMemoryLenght
        {
            get
            {
                return errorMemoryLenght;
            }
            set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException("errorMemoryLenght", "ErrorMemoryLenght cannot be lower than zero");
                errorMemoryLenght = value;
                errors = Vector<double>.Build.Dense(value);
            }
        }

        protected double momentum = 0.1;

        /// <summary>
        /// The idicator of how much previous gradients affect curently applied one.
        /// </summary>
        public double Momentum
        {
            get
            {
                return momentum;
            }
            set
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException("momentum", "Momentum cannot be lower than zero");
                momentum = value;
            }
        }

        protected double accuracy = 10;

        /// <summary>
        /// The indicator of how accurate the calculations are. For higher values the network will be able to get closer to the local minimum at a cost of learning speed.
        /// </summary>
        public double Accuracy
        {
            get
            {
                return accuracy;
            }
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("accuracy", "Accuracy must be greater than zero");
                accuracy = value;
            }
        }



        protected int exampleBlockSize = 5;
        /// <summary>
        /// The number of input-outout examples used to calculate the gradient at one time.
        /// </summary>
        public int ExampleBlockSize
        {
            get
            {
                return exampleBlockSize;
            }
            set
            {
                if (value < 1)
                {
                    throw new ArgumentOutOfRangeException("exampleBlockSize");
                }
                exampleBlockSize = value;
                learningData = new Matrix<double>[2, exampleBlockSize];
            }
        }

        protected Queue requests = Queue.Synchronized(new Queue());

        /// <summary>
        /// The function used to extract data from outside source.
        /// </summary>
        public Func<LearningBrain, (Matrix<double>, Matrix<double>)> DataExtractor { get; set; }
        /// <summary>
        /// The function to be invoked after each done block.
        /// </summary>
        public Action<LearningBrain> BlockDoneAction { get; set; }

        /// <param name="inputNeurons">Number of input parameters.</param>
        /// <param name="hiddenNeurons">Number of neurons on the hidden layer.</param>
        /// <param name="outputNeurons">Number of output neurons.</param>
        public LearningBrain(int inputNeurons, int hiddenNeurons, int outputNeurons) : base(inputNeurons, hiddenNeurons, outputNeurons)
        {
            SetupLearningBrain();
        }

        /// <param name="inputNeurons">Number of input parameters.</param>
        /// <param name="hiddenNeurons">Array containing number of neurons on each hidden layer.</param>
        /// <param name="outputNeurons">Number of output neurons.</param>
        public LearningBrain(int inputNeurons, int[] hiddenNeurons, int outputNeurons) : base(inputNeurons, hiddenNeurons, outputNeurons)
        {
            SetupLearningBrain();
        }

        /// <summary>
        /// Uses already existing brain to build this instance of LearningBrain.
        /// </summary>
        /// <param name="brainCore">Brain to be used as a base.</param>
        public LearningBrain(BrainCore brainCore) : base(brainCore)
        {
            SetupLearningBrain();
        }

        /// <summary>
        /// Uses already existing brain to build this instance of LearningBrain.
        /// </summary>
        /// <param name="learningBrain">Brain to be used as a base.</param>
        public LearningBrain(LearningBrain learningBrain) : base(learningBrain)
        {
            SetupLearningBrain();

        }


        protected void SetupLearningBrain()
        {
            errors = Vector<double>.Build.Dense(errorMemoryLenght);
            synapsGradientMomentum = GetSynapsGradientFrame();
            biasGradientMomentum = GetBiasGradientFrame();
        }

        /// <summary>
        /// Teaches the BrainCore on a given number of exaples. Uses DataExtractor to extract learning data.
        /// </summary>
        /// <param name="count">The number of example blocks to learn.</param>
        /// <returns></returns>
        public double LearnBlocks(int count)
        {
            if (DataExtractor == null)
                throw new NullReferenceException("DataExtractor cannot be null");
            (Matrix<double>, Matrix<double>) data;
            for (int c = 0; c < count; c++)
            {
                for (int i = 0; i < exampleBlockSize; i++)
                {
                    data = DataExtractor(this);
                    learningData[0, i] = data.Item1;
                    learningData[1, i] = data.Item2;
                }
                CalculateOneBlock();
            }
            return MeanSquaredError;
        }

        /// <summary>
        /// Teaches the BrainCore on a given number of input and output exaples.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="expectedOutput"></param>
        /// <returns></returns>
        public double LearnBlocks(Matrix<double>[] input, Matrix<double>[] expectedOutput)
        {
            if (input.Length != expectedOutput.Length)
                throw new ArgumentException("Both input and expected output must be of the same lenght.");
            int limit = input.Length / exampleBlockSize + 1;
            int position = 0;
            for (int c = 0; c < limit; c++)
            {
                for (int i = 0; i < exampleBlockSize; i++)
                {
                    learningData[0, 1] = input[position];
                    learningData[1, 1] = expectedOutput[position];
                    position++;
                    if (position >= input.Length)
                        position = 0;
                }
                CalculateOneBlock();
            }
            return MeanSquaredError;
        }

        /// <summary>
        /// Teaches the BrainCore while conditionFunction returns true. Uses DataExtractor to extract learning data.
        /// </summary>
        /// <param name="conditionFunction">The function to be used as a loop condition. Arguments passed to it are respectively BlocksDone, ExamplesDone and MeanSquaredError.</param>
        /// <returns></returns>
        public double LearnBlocksWhile(Func<int, long, double, bool> conditionFunction)
        {
            while (conditionFunction(BlocksDone, ExamplesDone, MeanSquaredError))
            { 
                for (int i = 0; i < exampleBlockSize; i++)
                {
                    (Matrix<double>, Matrix<double>) data = DataExtractor(this);
                    learningData[0, 1] = data.Item1;
                    learningData[1, 1] = data.Item2;
                }
                CalculateOneBlock();
            }
            return MeanSquaredError;
        }

        /// <summary>
        /// Teaches the BrainCore on a given number of input and output exaples while the conditionFunction returns true.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="expectedOutput"></param>
        /// <param name="conditionFunction">The function to be used as a loop condition. Arguments passed to it are respectively BlocksDone, ExamplesDone and MeanSquaredError.</param>
        /// <returns></returns>
        public double LearnBlocksWhile(Matrix<double>[] input, Matrix<double>[] expectedOutput, Func<int, long, double, bool> conditionFunction)
        {
            if (conditionFunction == null)
                throw new ArgumentNullException("conditionFunction");
            if (input.Length != expectedOutput.Length)
                throw new ArgumentException("Both input and expected output must be of the same lenght.");
            int position = 0;
            while(conditionFunction(BlocksDone, ExamplesDone, MeanSquaredError))
            { 
                for (int i = 0; i < exampleBlockSize; i++)
                {
                    learningData[0, 1] = input[position];
                    learningData[1, 1] = expectedOutput[position];
                    position++;
                    if (position >= input.Length)
                        position = 0;
                }
                CalculateOneBlock();
            }
            return MeanSquaredError;
        }

        /// <summary>
        /// Calculates and applies gradients for each example in current block. Returns MeanSquaredError.
        /// </summary>
        protected double CalculateOneBlock()
        {
            gradientTouple = CalculateGradients(learningData[0, 0], learningData[1, 0]);
            synapsGradient = gradientTouple.Item1;
            biasGradient = gradientTouple.Item2;
            errors[lastErrorPosition] = gradientTouple.Item3;
            MeanSquaredError = errors.Sum() / errorMemoryLenght;
            lastErrorPosition++;
            lastErrorPosition %= errorMemoryLenght;

            for (int i = 1; i < exampleBlockSize; i++)
            {
                gradientTouple = CalculateGradients(learningData[0,i], learningData[1,i]);
                for (int l = 0; l < TotalSynapsLayers; l++)
                {
                    synapsGradient[l] += gradientTouple.Item1[l];
                    biasGradient[l] += gradientTouple.Item2[l];
                }
            }

            for (int l = 0; l < TotalSynapsLayers; l++)
            {
                synapsGradient[l] /= (double)exampleBlockSize;
                synapsGradientMomentum[l] = synapsGradient[l] / accuracy + synapsGradientMomentum[l] * momentum;

                biasGradient[l] /= (double)exampleBlockSize;
                biasGradientMomentum[l] = biasGradient[l] / accuracy + biasGradientMomentum[l] * momentum;
            }
            ApplyGradients(synapsGradientMomentum, biasGradientMomentum);
            BlocksDone++;
            BlockDoneAction?.Invoke(this);
            ExamplesDone += exampleBlockSize;
            return MeanSquaredError;
        }


    }
}

