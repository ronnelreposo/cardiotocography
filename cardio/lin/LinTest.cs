using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Collections.Generic;

namespace lin
{
    [TestClass]
    public class LinTest
    {
        Func<int, int> fTimes2 = x => x * 2;
        IEnumerable<int> expectedTimes2 = new[] { 1, 2, 3 }.Select(x => x * 2);
        int[] vector = { 1, 2, 3 };
        double[] vectorA = { 1, 2, 3 };
        double[] vectorB = { 2, 3, 4 };
        double[][] matrix = {
            new double [] { 1, 2, 3 },
            new double [] { 2, 3, 4 },
            new double [] { 1, 2, 3 }
        };

        [TestMethod]
        public void TestVectorOperation ()
        {
            var acc = new int[vector.Length];
            var times2 = vector.VectorOperation(fTimes2, acc);

            CollectionAssert.AreEqual(expectedTimes2.ToArray(), times2);
        }

        [TestMethod]
        public void TestVectorOperation2()
        {
            var vecTimes2 = vector.VectorMap(fTimes2);
            Func<int, int, int> add = (x, y) => x + y;
            var output = vector.VectorOperation2(vecTimes2, add, new int[vecTimes2.Length]);
            var expected = vector.Zip(vecTimes2, (x, y) => x + y).ToArray();

            CollectionAssert.AreEqual(output, expected);
        }

        [TestMethod]
        public void TestVectorMap()
        {
            var times2 = vector.VectorMap(fTimes2);
            
            CollectionAssert.AreEqual(expectedTimes2.ToArray(), times2);
        }

        [TestMethod]
        public void TestVectorMul()
        {
            var output = vectorA.VectorMul(vectorB);
            var expected = new double[] { 2, 6, 12 };

            CollectionAssert.AreEqual(expected, output);
        }

        [TestMethod]
        public void TestVectorAdd()
        {
            var output = vectorA.VectorAdd(vectorB);
            var expected = new double[] { 3, 5, 7 };
            CollectionAssert.AreEqual(output, expected);
        }

        [TestMethod]
        public void TestVectorDotProduct()
        {
            var output = vectorA.VectorDotProduct(vectorB);
            const double expected = 20;

            Assert.AreEqual(output, expected);
        }

        [TestMethod]
        public void TestVectorDotProductMatrix()
        {
            var aDot = vectorA.VectorDotProduct(vectorA);
            var bDot = vectorA.VectorDotProduct(vectorB);
             
            var output = vectorA.VectorDotProductMatrix(matrix);
            var expected = new double[] { aDot, bDot, aDot };

            CollectionAssert.AreEqual(output, expected);
        }
    }
}