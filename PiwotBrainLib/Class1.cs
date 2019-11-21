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
            BrainCore bc = new BrainCore(2, 1, 4);
            bc.SaveToFile("", "1");
            bc.ExpandLayer(1, 1);
            bc.SaveToFile("", "2");
        }
    }
}
