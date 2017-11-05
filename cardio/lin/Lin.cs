using System;

namespace LinTest
{
    /// <summary>
    /// Represents Linear Algebra Vector functions.
    /// </summary>
    /// <typeparam name="T">Input Type.</typeparam>
    /// <typeparam name="V">Output Type.</typeparam>
    public static class Lin<T, V>
    {
        /// <summary>
        /// Vector operation. (This method is deprecated, use Select in Linq instead.)
        /// </summary>
        /// <param name="f">The operation to be performed in each element.</param>
        /// <param name="i">index</param>
        /// <param name="acc">The Accumulator vector</param>
        /// <param name="xs">Input vector.</param>
        /// <returns>The accumulated vector.</returns>
        public static V[] opxs (Func<T, V> f, int i, V[] acc, T[] xs)
        {
            if ( i > ( xs.Length - 1 ) ) { return acc; }
            else
            {
                acc[i] = f(xs[i]);
                return opxs(f, ( i + 1 ), acc, xs);
            }
        } /* end opxs. */

        /// <summary>
        /// Operation on two vectors using function (f). (This can achived also by using [Zip] in Linq.)
        /// </summary>
        /// <param name="mapper">The mapper function.</param>
        /// <param name="i">index.</param>
        /// <param name="acc">Vector accumulator.</param>
        /// <param name="xs">First Vector.</param>
        /// <param name="ys">Second Vector.</param>
        /// <returns></returns>
        public static V[] opxs2 (Func<T, T, V> mapper, int i, V[] acc, T[] xs, T[] ys)
        {
            if ( xs.Length != ys.Length )
            {
                throw new Exception("Vectors are not in same length.");
            }

            if ( i > ( xs.Length - 1 ) ) { return acc; }
            else
            {
                acc[i] = mapper(xs[i], ys[i]);
                return opxs2(mapper, ( i + 1 ), acc, xs, ys);
            }
        } /* end opxs2. */

        /// <summary>
        /// Projects a vector to another form. (This can achived also by using [Select] in Linq.)
        /// </summary>
        /// <param name="mapper">The mapper function.</param>
        /// <param name="vector">The input vector.</param>
        /// <returns></returns>
        public static V[] mapxs (Func<T, V> mapper, T[] vector) => opxs(mapper, 0, new V[vector.Length], vector);

    } /* end class. */
} /* end namespace. */