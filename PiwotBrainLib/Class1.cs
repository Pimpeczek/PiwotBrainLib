using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PiwotBrainLib
{
    public static class Class1
    {
        static void Main(string[] args)
        {
            BrainCore bc = new BrainCore(4, 2, 4);
            bc.SaveToFile("", "1");
            bc.StreachLayer(0, 2, 2, 2, 1);
            bc.SaveToFile("", "2");
            Console.ReadKey(true);
        }
    }
}
