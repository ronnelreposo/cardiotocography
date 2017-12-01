using System.Windows.Controls;

namespace cardio.Ext
{
    /// <summary>
    /// Represents Button Extension
    /// </summary>
    static class ButtonExt
    {
        /// <summary>
        /// Disables the button
        /// </summary>
        /// <param name="button">Given button to be disbaled</param>
        /// <returns>button</returns>
        internal static Button Disable (this Button button)
        {
            if ( !button.IsEnabled ) return button;

            button.IsEnabled = false;

            return button;
        }

        /// <summary>
        /// Enables the button
        /// </summary>
        /// <param name="button">Given button to enabled</param>
        /// <returns>button</returns>
        internal static Button Enable (this Button button)
        {
            if ( button.IsEnabled ) return button;

            button.IsEnabled = true;

            return button;
        }
    }
}
