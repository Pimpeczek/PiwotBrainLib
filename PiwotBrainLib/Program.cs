using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using System.IO;

namespace PiwotBrainLib
{
    class Program
    {
        static bool doGo;
        static bool foofoo;
        static bool doFinish;
        static int hAxis = 0;
        static int vAxis = 99;
        static Random rng;
        static Stopwatch totalTime;
        static LearningBrain l;
        static void Main(string[] args)
        {
            Console.SetWindowSize(200, 63);
            Console.CursorVisible = false;
            Console.OutputEncoding = Encoding.UTF8;
            rng = new Random(DateTime.Now.Millisecond);
            Thread t = new Thread(AsyncInputChecker);
            t.Start();
            totalTime = new Stopwatch();
            totalTime.Start();
            doFinish = false;
            doGo = true;
            foofoo = false;

            l = new LearningBrain(1, new int[] { 12 }, 1)
            {
                DataExtractor = (i) => 
                {
                    double x = rng.NextDouble();
                    double r = rng.NextDouble();
                    double z = rng.NextDouble();
                    return (Matrix<double>.Build.Dense(1, 1, (rows, cols)=> x), Matrix<double>.Build.Dense(1, 1, FuncToLearn(x)));
                },
                ExampleBlockSize = 2,
                Accuracy = 10
                
            };
            Stopwatch sw = new Stopwatch();
            l.SaveToFile(Directory.GetCurrentDirectory(), "musk");
            BrainCore b = new BrainCore("musk.txt");
            do {
                sw.Restart();
                if (doGo)
                {
                    DrawFunc(l);
                    l.LearnBlocks(5000);
                }
                else
                {
                    DrawFunc(b);
                    Thread.Sleep(50);
                }
                
                sw.Stop();
                Console.WriteLine($"{l.BlocksDone}".PadRight(20));
                Console.WriteLine($"{sw.ElapsedMilliseconds}".PadRight(20));
            } while(true);

        }

        static void AsyncInputChecker()
        {
            ConsoleKey key;
            do
            {
                key = Console.ReadKey(true).Key;
                switch(key)
                {
                    case ConsoleKey.Spacebar:
                        doGo = !doGo;
                        break;
                    case ConsoleKey.LeftArrow:
                        hAxis--;
                        break;
                    case ConsoleKey.RightArrow:
                        hAxis++;
                        break;
                    case ConsoleKey.DownArrow:
                        vAxis--;
                        break;
                    case ConsoleKey.UpArrow:
                        vAxis++;
                        break;
                    case ConsoleKey.S:
                        if (!doGo)
                            l.SaveToFile(Directory.GetCurrentDirectory(), "musk");
                        break;
                }

            } while (key != ConsoleKey.Escape);
        }
        static double FuncToLearn(double x)
        {
            return (Math.Sin(Math.PI * 2 * (x-0.25)) + 1) / 2;
            //return (Math.Sin(Math.PI * 4 * (x)) + 1) / 2;
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
            //return x;
        }

        static double FuncToLearn(double x, double y)
        {

            return ((x + y) * 2 % 2) / 2;
        }

        static double FuncToLearn(double x, double y, double z)
        {
            x = ((x + y) * 2 % 2) / 2;
            return x > z ? z : x;
        }

        static void LearnFunction(BrainCore b)
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

        static void DrawFunc(BrainCore b)
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
                //v2[1] = ((double)hAxis / 100.0) % 1;
                //v2[2] = ((double)vAxis / 100.0) % 1;
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
    }
}
