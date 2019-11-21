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
            BrainCore bc = new BrainCore(3, 2, 4);
            bc.SaveToFile("", "1");
            bc.StreachLayer(1, 2);
            bc.SaveToFile("", "2");
        }
    }
}
