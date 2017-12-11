/// Licensed to Ronnel Reposo under one or more agreements.
/// Ronnel Reposo licenses this file to you under the MIT license.
/// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using static System.Diagnostics.Contracts.Contract;

namespace lineal
{
    /// <summary>
    /// Represents Linear Algebra Vector functions.
    /// </summary>
    public static class Vector
    {
        /// <summary>
        /// Vector operation.
        /// </summary>
        /// <typeparam name="T">Input Type</typeparam>
        /// <typeparam name="V">Output Type</typeparam>
        /// <param name="xs">Input vector.</param>
        /// <param name="f">The operation to be performed in each element.</param>
        /// <param name="acc">The Accumulator vector</param>
        /// <param name="i">index initialized to 0.</param>
        /// <returns>The accumulated vector.</returns>
        public static V[] VectorOperation<T,V> (this T[] vector, Func<T, V> f, V[] acc, int i = 0)
        {
            if ( i > ( vector.Length - 1 ) ) { return acc; }
            else
            {
                acc[i] = f(vector[i]);
                return vector.VectorOperation(f, acc, ( i + 1 ));
            }
        } /* end VectorOperation. */

        /// <summary>
        /// Operation on two vectors using function (f).
        /// </summary>
        /// <typeparam name="T">Input Type</typeparam>
        /// <typeparam name="V">Output Type</typeparam>
        /// <param name="firstvector">First Vector.</param>
        /// <param name="secondVector">Second Vector.</param>
        /// <param name="mapper">The mapper function.</param>
        /// <param name="acc">Vector accumulator.</param>
        /// <param name="i">index initialized to 0.</param>
        /// <returns>The accumulated Vector.</returns>
        public static V[] VectorOperation2<T, V> (this T[] firstvector, T[] secondVector, Func<T, T, V> mapper, V[] acc, int i = 0)
        {
            if ( firstvector.Length != secondVector.Length )
            {
                throw new Exception("Vectors are not in same length.");
            }

            if ( i > ( firstvector.Length - 1 ) ) { return acc; }
            else
            {
                acc[i] = mapper(firstvector[i], secondVector[i]);
                return firstvector.VectorOperation2(secondVector, mapper, acc, ( i + 1 ));
            }
        } /* end VectorOperation2. */

        /// <summary>
        /// Projects a vector to another form.
        /// </summary>
        /// <typeparam name="T">Input Type</typeparam>
        /// <typeparam name="V">Output Type</typeparam>
        /// <param name="vector">The input vector.</param>
        /// <param name="mapper">The mapper function.</param>
        /// <returns>The accumulated projected vector.</returns>
        public static V[] VectorMap<T, V> (this T[] vector, Func<T, V> mapper)
        {
            /* Use Linq Select for code optimization. */
            return vector.Select(mapper).ToArray();
        }

        /// <summary>
        /// Vector Multiplication.
        /// </summary>
        /// <param name="firstVector">First Vector.</param>
        /// <param name="secondVector">Second Vector.</param>
        /// <returns>new Vector Product.</returns>
        public static double[] VectorMul (this double[] firstVector, double[] secondVector)
        {
            Func<double, double, double> mul = (x, y) => x * y;

            /* Use Linq Zip for code optimization. */
            return firstVector.Zip(secondVector, mul).ToArray();
        }

        /// <summary>
        /// Vector Addtion.
        /// </summary>
        /// <param name="firstVector">First Vector.</param>
        /// <param name="secondVector">Second Vector.</param>
        /// <returns>new Sum Vector.</returns>
        public static double[] VectorAdd (this double[] firstVector, double[] secondVector)
        {
            Func<double, double, double> add = (x, y) => x + y;

            /* Use Linq Zip for code optimization. */
            return firstVector.Zip(secondVector, add).ToArray();
        }

        /// <summary>
        /// The Dot Product of two Vectors.
        /// </summary>
        /// <param name="firstVector">First Vector.</param>
        /// <param name="secondVector">Second Vector.</param>
        /// <returns>The Dot Product.</returns>
        public static double VectorDotProduct (this double[] firstVector, double[] secondVector)
        {
            Requires(firstVector != null, "First Vector (xs) is not null.");
            Requires(secondVector != null, "Second Vector (ys) is not null.");

            /* Use Linq Sum for code optimization. */
            return firstVector.VectorMul(secondVector).Sum();
        }

        /// <summary>
        /// Determines the Dot Product of a given vector to each vector in a given matrix,
        /// and returns vector of accumulated Dot Product Vector.
        /// </summary>
        /// <param name="vector">The given Vector.</param>
        /// <param name="matrix">The given Matrix.</param>
        /// <param name="acc">The Vector accumulator.</param>
        /// <param name="i">index initialized to 0.</param>
        /// <returns>The accumulated Dot Product Vector.</returns>
        static double[] VectorDotProductMatrix (this double[] vector, double[][] matrix, double[] acc, int i = 0)
        {
            /* Scape code contracts. ---- */

            if ( i > ( matrix.Length - 1 ) ) { return acc; }

            acc[i] = vector.VectorMul(matrix[i]).Sum();

            return vector.VectorDotProductMatrix(matrix, acc, ( i + 1 ));
        }

        /// <summary>
        /// Determines the Dot Product of a given vector to each vector in a given matrix,
        /// and returns vector of Dot Product Vector.
        /// </summary>
        /// <param name="vector">The given Vector.</param>
        /// <param name="matrix">The given Matrix.</param>
        /// <returns>The accumulated Dot Product Vector.</returns>
        public static double[] VectorDotProductMatrix (this double[] vector, double[][] matrix)
        {
            Requires(vector != null, "The given vector should must not be null.");
            Requires(matrix != null, "The given matrix should must not be null.");

            return vector.VectorDotProductMatrix(matrix, new double[matrix.Length]);
        }

    } /* end class. */
}
