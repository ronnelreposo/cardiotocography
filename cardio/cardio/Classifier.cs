using System;
using System.Linq;
using static System.Math;
using lin;

namespace cardio
{
    /// <summary>
    /// Represents as Classifier Engine.
    /// </summary>
    internal static class Classifier
    {
        /// <summary>
        /// Weighted Sum with Bias.
        /// </summary>
        /// <param name="inputVector">Input Vector.</param>
        /// <param name="weightsMatrix">Weights Matrix.</param>
        /// <param name="biasVector">Bias Vector.</param>
        /// <returns>Weigted Sum Vector.</returns>
        static double[] biasedWeightedSum (double[] inputVector, double[][] weightsMatrix, double[] biasVector) =>
            biasVector.VectorAdd(inputVector.VectorDotProductMatrix(weightsMatrix));

        /// <summary>
        /// Classifier Engine.
        /// </summary>
        /// <param name="inputVector">The Input Vector.</param>
        /// <returns>Output Class Vector.</returns>
        static double[] classify (double[] inputVector)
        {
            var firstHiddenLayerWeights = new double[][]
            {
                new[] { -0.535683,-0.141191,0.113704,-0.280238,-0.513372,0.118896,0.118266 },
                new[] { -0.884773,3.483094,0.256009,0.547940,-0.887624,-0.260597,-1.507266 },
                new[] { -0.172401,0.868466,0.515647,-0.206276,1.567889,0.154666,0.741788 },
                new[] { 0.347045,0.213125,0.563421,0.019287,0.294708,0.580635,0.409552 },
                new[] { -0.399822,0.159410,0.362866,0.070373,2.075633,-0.093059,1.036481 },
                new[] { 0.636117,-1.385243,-0.046540,0.138753,-0.241373,0.385755,0.285293 },
                new[] { 0.336681,-0.114951,0.329618,-0.274353,-0.602820,0.449632,-0.179750 },
                new[] { -0.677270,2.246282,0.290830,0.416275,1.153468,-0.083218,0.042591 },
                new[] { 0.366223,0.087092,0.042139,0.332673,0.544853,0.425909,0.679988 },
                new[] { 0.289060,0.209365,0.135112,0.031238,0.063794,0.036886,0.736910 }
            };
            var firstHiddenLayerBias = new[] { -1.183110, 0.215782, -0.072329, 0.421725, -0.027928, -0.659639, -1.694621, -0.095888, -1.114419, 0.025710 };
            var firstHiddenWeightedSum = biasedWeightedSum(inputVector, firstHiddenLayerWeights, firstHiddenLayerBias);
            var firstHiddenNetOutputs = firstHiddenWeightedSum.Select(Tanh).ToArray();

            var secHiddenLayerWeights = new double[][]
            {
                new[] { 0.060251,-0.410701,0.334272,-0.304857,0.425623,0.459251,0.885662,0.319826,0.748129,0.004529 },
                new[] { -0.190495,-0.649993,-0.297271,-0.079893,0.028677,0.424193,0.279423,-0.715746,0.013316,0.590390 },
                new[] { 0.625548,1.707777,-1.340614,-0.160406,-1.838768,-0.128937,0.298701,-0.730869,-0.653966,-0.286304 },
                new[] { 0.103474,0.241207,0.289159,-0.297770,-0.109335,0.592422,1.039788,0.021677,0.840314,0.753864 },
                new[] { 0.742126,-0.221056,-0.130290,0.038440,0.255014,0.932129,0.790503,-0.156448,0.474562,0.206995 },
                new[] { 1.041067,-0.075909,-0.318840,0.201361,-0.534134,0.493703,0.920956,-0.101970,0.185884,0.099387 },
                new[] { 0.461124,-0.316137,0.165494,-0.476027,0.255663,0.788544,0.323238,-0.454547,0.226956,-0.497691 },
                new[] { 0.566055,2.754974,0.934421,0.149694,0.858572,-0.844048,-0.023037,1.911453,-0.134889,0.131041 },
                new[] { 0.928492,-0.084657,0.346598,0.320652,-0.042574,1.008584,1.049624,0.253614,0.025321,0.180770 },
                new[] { 0.640074,-1.557197,0.073825,0.493217,0.574084,0.710434,0.057439,-0.727300,0.127141,0.366996 },
            };
            var secHiddenLayerBias = new[] { 0.129653, 0.511955, 0.307257, 0.378682, 0.209885, -0.294118, -0.549358, 0.163794, -0.324625, 0.485924 };
            var secHiddenWeightedSum = biasedWeightedSum(firstHiddenNetOutputs, secHiddenLayerWeights, secHiddenLayerBias);
            var secHiddenNetOutputs = secHiddenWeightedSum.Select(Tanh).ToArray();

            var outputLayerWeights = new double[][]
            {
                new[] { 0.355524,0.667619,1.048840,0.152452,0.252030,0.619544,0.084857,-0.443258,0.491056,0.098885 },
                new[] { -0.065105,0.513361,2.066656,0.789753,0.609331,0.851257,0.074146,1.666547,0.074938,-0.198328 },
                new[] { 0.784767,0.393893,0.880301,0.199494,0.290644,0.636244,0.557407,-0.518559,0.057453,-0.129881 },
                new[] { 0.010491,-0.400490,0.939674,0.173190,0.635006,0.441717,0.823339,1.029103,0.893646,-0.062708 },
                new[] { 0.010667,0.036595,0.670503,0.212975,0.536869,0.444933,0.504348,-0.452175,0.405928,0.584689 },
                new[] { 0.782975,0.073682,-1.434274,0.685647,0.122003,0.440585,0.445463,1.411089,0.256008,-0.948879 },
                new[] { -0.338009,0.147413,-0.964443,-0.294292,0.138106,0.571631,0.925734,0.050965,0.820055,0.667740 },
                new[] { 0.867212,0.507087,-0.940812,0.281326,0.522925,0.603955,0.516550,0.164695,0.430572,1.408784 },
                new[] { 0.787742,0.630823,0.636885,0.556916,0.026780,0.239597,0.668234,-0.661122,0.334565,0.337981 },
                new[] { 0.271870,0.512810,0.733583,-0.048812,0.711727,0.638616,0.096019,-1.476907,0.426373,0.022986 },
                new[] { 0.025441,-0.193596,0.634286,0.241555,0.283347,-0.161585,-0.231059,1.698397,-0.249537,-0.544146 },
                new[] { 0.487712,0.674506,0.691306,0.483758,0.095772,0.638273,0.666845,-1.079267,-0.060443,0.474487 },
                new[] { 0.815792,0.931031,-0.179969,0.657107,0.450946,0.103969,-0.027498,-0.190065,0.086145,1.419796 },
            };
            var outputBias = new[] { 0.520434, 0.444391, 0.284465, 0.265100, 0.273132, 0.148382, 0.685081, 0.702207, 0.352506, 0.304391, 0.630367, 0.881243, 0.220049 };
            var outputWeigtedSum = biasedWeightedSum(secHiddenNetOutputs, outputLayerWeights, outputBias);
            return outputWeigtedSum.Select(Tanh).ToArray();
        } /* end classifier method. */

        /// <summary>
        /// Classifier Engine Facade.
        /// </summary>
        /// <param name="percentConv">Percent Value Converter.</param>
        /// <param name="inputVector">The InputVector</param>
        /// <returns>Output Class Vector.</returns>
        internal static double[] Classify (Func<double, double> percentConv, double[] inputVector) => classify(inputVector).VectorMap(percentConv);
    } /* end class. */
} /* end namespace. */