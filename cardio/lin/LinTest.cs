using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Collections.Generic;

namespace lin
{
    [TestClass]
    public class LinTest
    {
        IEnumerable<int> expectedTimes2 = new[] { 1, 2, 3 }.Select(x => x * 2);
        int[] vector = new[] { 1, 2, 3 };

        [TestMethod]
        public void TestOpxs ()
        {
            var acc = new int[vector.Length];
            var times2 = Lin<int, int>.opxs(x => x * 2, 0, acc, vector);

            Equals(expectedTimes2, times2);
        }

        [TestMethod]
        public void TestMapxs()
        {
            var times2 = Lin<int, int>.mapxs(x => x * 2, vector);
            Equals(expectedTimes2, times2);
        }

        [TestMethod]
        public void TestOpxs2()
        {
            var times2 = Lin<int, int>.mapxs(x => x * 2, vector);
            var opxs2 = Lin<int, int>.opxs2((x, y) => x + y, 0, new int[times2.Length], vector, times2);
            var expected = vector.Zip(times2, (x, y) => x + y);

            Equals(opxs2, expected);
        }
    }
}
