using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;

namespace PiwotBrainLib
{
    class Program
    {
        static bool doGo;
        static bool foofoo;
        static bool doFinish;
        static Random rng;
        static Stopwatch totalTime;
        static void Main(string[] args)
        {
            Console.SetWindowSize(200, 63);
            Console.CursorVisible = false;
            Console.OutputEncoding = Encoding.UTF8;
            rng = new Random(DateTime.Now.Millisecond);
            Thread t = new Thread(ToggleClaculation);
            t.Start();
            totalTime = new Stopwatch();
            totalTime.Start();
            doFinish = false;
            doGo = true;
            foofoo = false;

            Brain b = new Brain(1, new int[] { 4,8}, 1);

            Learner l = new Learner()
            {
                Brain = b,
                DataExtractor = (i, j) => { double x = rng.NextDouble(); return (Matrix<double>.Build.Dense(1, 1, x), Matrix<double>.Build.Dense(1, 1, FuncToLearn(x))); },
                ExampleBlockSize = 1,
                Accuracy = 10
                
            };
            l.LearnToGivenError(0.00005);
            do
            {
                DrawFunc(b);
            }while(true);
        }
        static double FuncToLearn(double x)
        {
            return (Math.Sin(Math.PI * 4 * (x)) + 1) / 2;
            //return x * x;
            //return x * 16 % 2 > 1 ? 1 : 0;
            /*if(x > 0.5)
            {
                return x > 0.5625 ? 1 / (32 * (x - 0.5)) + 0.5 : 1;
            }
            else if(x < 0.5)
            {
                return x < 0.4375 ? 1 / (32 * (x - 0.5)) + 0.5 : 0;
            }
            else
            {
                return x;
            }*/
            //return (x * 10 % 2) / 2;
        }

        static void LearnFunction(Brain b)
        {
            long counter = 0;
            int blockCounter = 0;
            int areaDiffPrec = 2048;
            int savedCostChanges = 5;
            double learBlocks = 10;
            double learnedBlocks = 0;
            Vector<double> averageCostChange = Vector<double>.Build.Dense(savedCostChanges, 0);
            double lastCost = 1;

            Stopwatch pointTime = new Stopwatch();
            Stopwatch drawingTime = new Stopwatch();
            double x = 0, sinX = 0;
            double areaDiff = 0, bestAreaDiff = 10000;
            Vector<double> v1 = Vector<double>.Build.Dense(1);
            Vector<double> v2 = Vector<double>.Build.Dense(1);
            Vector<double> vt;

            do
            {
                if (doGo)
                {
                    pointTime.Restart();
                    learnedBlocks = 0;
                    do
                    {


                        for (int i = 0; i < learBlocks; i++)
                        {
                            x = (rng.NextDouble());
                            sinX = FuncToLearn(x);

                            v1[0] = x;
                            v2[0] = sinX;
                            //b.LearnOne(v1, v2);

                        }
                        learnedBlocks++;

                    } while (pointTime.ElapsedMilliseconds < 500);
                    pointTime.Stop();
                    drawingTime.Restart();
                    counter += (int)(learnedBlocks * learBlocks);
                    DrawFunc(b);
                    //b.DrawBrain(160, 0);
                    areaDiff = Vector<double>.Build.Dense(areaDiffPrec, (y) => { v1[0] = y / (double)areaDiffPrec; return Math.Abs(b.Calculate(v1)[0] - FuncToLearn(v1[0])) / (double)areaDiffPrec; }).Sum();
                    if (bestAreaDiff > areaDiff)
                        bestAreaDiff = areaDiff;

                    averageCostChange[blockCounter % savedCostChanges] = areaDiff - lastCost;
                    lastCost = areaDiff;
                    blockCounter++;
                    Console.SetCursorPosition(0, 60);
                    Console.WriteLine($"Iteration: {counter}, Difference area: {areaDiff.ToString("0.####")}, Best difference area: {bestAreaDiff.ToString("0.######")}, Avg. cost change: {averageCostChange.Sum() / savedCostChanges}".PadRight(150));
                    Console.WriteLine($"Time: {totalTime.Elapsed}, Iterations/s: {learnedBlocks * learBlocks / pointTime.ElapsedMilliseconds * 1000.0}    ");
                    drawingTime.Stop();
                }
                else
                {
                    Thread.Sleep(100);
                }

            } while (!doFinish);
        }

        static void DrawFunc(Brain b)
        {
            Vector<double> v2 = Vector<double>.Build.Dense(1);
            Console.SetCursorPosition(0, 0);
            int sizex = 160, sizey = 60;
            int xpos;
            bool xposFound;
            string str;
            double t;
            double[] values = new double[sizex];
            for (int i = 0; i < sizex; i++)
            {
                v2[0] = ((double)i / sizex);
                values[i] = b.Calculate(v2)[0] * sizey;
                //values[i] = FuncToLearn((double)i / sizex) * sizey;
            }

            for (int i = 0; i < sizey; i++)
            {
                str = "";
                t = sizey - i;
                xpos = 0;
                xposFound = false;
                for (int j = 0; j < sizex; j++)
                {
                    if (values[j] >= t)
                    {
                        str += '█';
                    }
                    else if (values[j] >= t - 0.5)

                    {
                        str += '▄';
                    }
                    else
                    {
                        str += ' ';
                    }
                }
                Console.WriteLine(str);
            }
        }
        static void ToggleClaculation()
        {
            ConsoleKey key;
            do
            {
                key = Console.ReadKey(true).Key;
                if (key == ConsoleKey.Spacebar)
                {
                    doGo = !doGo;

                    if (doGo)
                    {
                        totalTime.Start();
                        foofoo = !foofoo;
                    }
                    else
                        totalTime.Stop();
                }
                if (key == ConsoleKey.Escape)
                {
                    doFinish = true;
                }
            } while (true);
        }
    }
}
