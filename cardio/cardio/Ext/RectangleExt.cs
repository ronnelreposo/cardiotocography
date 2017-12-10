using System.Windows.Shapes;
using static System.Diagnostics.Contracts.Contract;

namespace cardio.Ext
{
    /// <summary>
    /// Represents a Rectangle Extension
    /// </summary>
    static class RectangleExt
    {
        /// <summary>
        /// Change Width Wrapper.
        /// </summary>
        /// <param name="rec">Rectangle (Control).</param>
        /// <param name="width">Width of Rectangle.</param>
        /// <returns>Rectangle that has changed its width.</returns>
        internal static Rectangle ChangeWidth (this Rectangle rec, double width)
        {
            Requires(rec != null);

            rec.Width = width;

            return rec;
        }

        /// <summary>
        /// Resets the Width of the Rectrangle to default width.
        /// </summary>
        /// <param name="rec">the given rectangle.</param>
        /// <param name="defaultWidth">initialized to 0.</param>
        /// <returns>resetted rectangle.</returns>
        internal static Rectangle ResetWidth (this Rectangle rec, double defaultWidth = 0)
        {
            Requires(rec != null);

            rec.Width = defaultWidth;

            return rec;
        }
    }
}
