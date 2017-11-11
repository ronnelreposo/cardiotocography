using cardio.Ext;
using FirstFloor.ModernUI.Windows.Controls;
using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Shapes;
using static cardio.Classifier;
using static System.Array;
using static System.Convert;
using static System.Math;
using static System.Reactive.Linq.Observable;
using static System.Threading.Tasks.Task;

namespace cardio
{
    public partial class MainWindow : ModernWindow
    {
        public MainWindow ()
        {
            InitializeComponent();

            setUpReactiveEngine();
        } /* end constructor. */

        void setUpReactiveEngine ()
        {
            var sClassifyClicked = from evt in FromEventPattern(classify_button, "Click")
                                   select evt.Sender as Button;

            var sClickDisabledButon = from button in sClassifyClicked
                                      let fhrSelected = fhrClass_cb.IsChecked == true
                                      let nspSelected = nspClass_cb.IsChecked == true
                                      where fhrSelected || nspSelected
                                      select disableButton(button);

            var sClickToClassifyOutputRecord = from button in sClickDisabledButon
                                               let PlaceVal = 3
                                               let inputControlVector =
                                                new[] { lb_tb, ac_tb, fm_tb, uc_tb, dl_tb, ds_tb, dp_tb }
                                               let outputControlVector =
                                                new[] {
                                                    a_rec, b_rec, c_rec, d_rec, e_rec, ad_rec, de_rec, ld_rec,
                                                    fs_rec, susp_rec, normal_rec, suspect_rec, phato_rec }
                                               let clearedOutputControlVector =
                                                from rectangle in outputControlVector
                                                select rectangle.ResetWidth()
                                               let normInputs = encodeInputs(inputControlVector, PlaceVal)
                                               let outputVector = computeVectorOutputs(inputVector: normInputs, placeVal: PlaceVal)
                                               select new
                                               {
                                                   Button = button,
                                                   InputControlsVector = inputControlVector,
                                                   OutputControlsVector = clearedOutputControlVector.ToArray(),
                                                   OutputVector = outputVector
                                               };

            var sClickToFhr = from record in sClickToClassifyOutputRecord
                              let fhrIsChecked = fhrClass_cb.IsChecked == true
                              let fhrClass = 10
                              where fhrIsChecked
                              select new
                              {
                                  Button = record.Button,
                                  ProgressedOutputVector = record
                                    .OutputVector
                                    .Take(fhrClass)
                                    .Zip(record.OutputControlsVector.Take(fhrClass), Tuple.Create)
                                    .Select(progress)
                              };

            var sClickToNsp = from record in sClickToClassifyOutputRecord
                              let nspIsChecked = nspClass_cb.IsChecked == true
                              let fhrClass = 10
                              where nspIsChecked
                              select new
                              {
                                  Button = record.Button,
                                  ProgressedOutputVector = record
                                    .OutputVector
                                    .Skip(fhrClass)
                                    .Zip(record.OutputControlsVector.Skip(fhrClass), Tuple.Create)
                                    .Select(progress)
                              };

            sClickToFhr
                .Subscribe(async record =>
                    ( await WhenAll(record.ProgressedOutputVector) )
                    .ToObservable()
                    .Take(1)
                    .Subscribe(_ => enableButton(record.Button)));

            sClickToNsp
                .Subscribe(async record =>
                    ( await WhenAll(record.ProgressedOutputVector) )
                    .ToObservable()
                    .Take(1)
                    .Subscribe(_ => enableButton(record.Button)));

        } /* end setUpReactiveEngine. */

        /// <summary>
        /// Encodes TextBox Inputs.
        /// </summary>
        /// <param name="inputTexboxVector">Input TextBox vector.</param>
        /// <param name="placeVal">Place value.</param>
        /// <returns>encoded input vector.</returns>
        double[] encodeInputs (TextBox[] inputTexboxVector, int placeVal = 3)
        {
            var lb_convToSec = Round(minToSec(getDoubleValue(lb_tb)), placeVal);
            var lb_max = 5;
            var ac_max = 30;
            var fm_max = 600;
            var uc_max = 30;
            var dl_max = 20;

            return new[] {
                minmax(lb_convToSec, lb_max, 1),
                minmax(getDoubleValue(ac_tb), ac_max),
                minmax(getDoubleValue(fm_tb), fm_max),
                minmax(getDoubleValue(uc_tb), uc_max),
                minmax(getDoubleValue(dl_tb), dl_max),
                getDoubleValue(ds_tb),
                minmax(getDoubleValue(dp_tb), lb_max)
            };
        } /* end normInputs. */

        /// <summary>
        /// Computes the output vector.
        /// </summary>
        /// <param name="inputVector">The input vector.</param>
        /// <param name="placeVal">place value.</param>
        /// <returns>output vector.</returns>
        double[] computeVectorOutputs (double[] inputVector, int placeVal = 3)
        {
            Converter<double, double> roundAtPlaceVal = x => Round(x, placeVal);
            var roundedNormInputs = ConvertAll(inputVector, roundAtPlaceVal);
            Func<double, double> minmaxPercent = x => minmax(x, 1, -1) * 100;

            return Classify(minmaxPercent, roundedNormInputs);
        } /* end compute vector outputs. */

        Task<Rectangle> progress(Tuple<double, Rectangle> valueAndControl) =>
            ProgressAsync(ToInt16(valueAndControl.Item1), valueAndControl.Item2, EaseInOutCubic);

        /// <summary>
        /// Converts minute to second.
        /// </summary>
        /// <param name="min">minute value.</param>
        /// <returns>converted minutes value.</returns>
        double minToSec (double min) => min / 60;

        /// <summary>
        /// Normalize the value from 0 to 1.
        /// </summary>
        /// <param name="value">The given value to be normalized.</param>
        /// <param name="max">The given maximum value.</param>
        /// <param name="min">The given minimum value.</param>
        /// <returns>scaled/normalized value.</returns>
        double minmax (double value, double max, double min = 0) => ( min.Equals(0) ) ? ( value / max ) : ( ( value - min ) / ( max - min ) );

        /// <summary>
        /// Converts the value of textbox to double.
        /// </summary>
        /// <param name="tb">The given textbox.</param>
        /// <returns>the converted value.</returns>
        double getDoubleValue (TextBox tb)
        {
            try
            {
                var text = tb.Text;
                return text.Equals(string.Empty) ? 0 : ToDouble(text);
            }
            catch ( Exception ) { return 0; }
        }

        /// <summary>
        /// Ease In-Out Cubic Function.
        /// </summary>
        readonly Func<double, double> EaseInOutCubic = x => ( x < 0.5 ) ? 4 * x * x * x : ( x - 1 ) * ( 2 * x - 2 ) * ( 2 * x - 2 ) + 1;

        /// <summary>
        /// Progresses the output (Asyncronously).
        /// It uses Rectangle and its width to project the maximum value,
        /// delta as the change value in every step.
        /// </summary>
        /// <param name="max">The maximum output value.</param>
        /// <param name="rec">The Rectangle(Control) to adjusted by its width.</param>
        /// <param name="delta">The change function. (Used for animation).</param>
        /// <param name="i">The step index. Zero by default.</param>
        /// <param name="delay">The step delay. 3 seconds by default.</param>
        /// <returns>The changed rectangle width.</returns>
        async Task<Rectangle> ProgressAsync (int max, Rectangle rec, Func<double, double> delta, int i = 0, int delay = 3)
        {
            if ( i.Equals(max) ) { return rec; }
            else
            {
                var normi = minmax(i, max);
                var i_ = delta(normi);
                var width_ = i_ * max;
                var rec_ = rec.ChangeWidth(width_);

                await Delay(delay);

                return await ProgressAsync(max, rec_, delta, ( i + 1 ));
            }
        }

        Button disableButton (Button button)
        {
            button.IsEnabled = false;
            return button;
        }

        Button enableButton (Button button)
        {
            button.IsEnabled = true;
            return button;
        }
    }
}