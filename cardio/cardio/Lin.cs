using System;

namespace cardio
{
    internal static class Lin<T, V>
    {
        internal static V[] opxs(Func<T, V> f, int i, V[] acc, T[] xs)
        {
            if (i > (xs.Length - 1)) { return acc; }
            else
            {
                acc[i] = f(xs[i]);
                return opxs(f, (i + 1), acc, xs);
            }
        }

        internal static V[] opxs2(Func<T, T, V> f, int i, V[] acc, T[] xs, T[] ys)
        {
            if (xs.Length != ys.Length)
            {
                throw new Exception("Vectors are not in same length.");
            }

            if (i > (xs.Length - 1)) { return acc; }
            else
            {
                acc[i] = f(xs[i], ys[i]);
                return opxs2(f, (i + 1), acc, xs, ys);
            }
        }

        internal static V[] mapxs(Func<T, V> f, T[] vector)
            => opxs(f, 0, new V[vector.Length], vector);
    }
}
