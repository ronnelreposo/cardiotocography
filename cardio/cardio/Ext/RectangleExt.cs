using System.Windows.Shapes;

namespace cardio.Ext
{
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
            rec.Width = width;
            return rec;
        }

        internal static Rectangle ResetWidth(this Rectangle rec, double defaultWidth = 0)
        {
            rec.Width = defaultWidth;
            return rec;
        }
    }
}
