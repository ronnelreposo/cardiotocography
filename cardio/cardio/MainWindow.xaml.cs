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
        public MainWindow () {

            InitializeComponent();

            var sClassifyClicked = FromEventPattern(classify_button, "Click").Select(evt => ( evt.Sender as Button ));
            var sClickDisabledButon = sClassifyClicked.Select(button => { button.IsEnabled = false; return button; });
            sClickDisabledButon.Subscribe(async button =>
                {
                    const int PlaceVal = 3;
                    var lb_convToSec = Round(minToSec(getDoubleValue(lb_tb)), PlaceVal);
                    var lb_max = 5;
                    var ac_max = 30;
                    var fm_max = 600;
                    var uc_max = 30;
                    var dl_max = 20;
                    var normInputs = new[] {
                        minmax(lb_convToSec, lb_max, 1),
                        minmax(getDoubleValue(ac_tb), ac_max),
                        minmax(getDoubleValue(fm_tb), fm_max),
                        minmax(getDoubleValue(uc_tb), uc_max),
                        minmax(getDoubleValue(dl_tb), dl_max),
                        getDoubleValue(ds_tb),
                        minmax(getDoubleValue(dp_tb), lb_max)
                    };
                    Converter<double, double> roundAtPlaceVal = x => Round(x, PlaceVal);
                    var roundedNormInputs = ConvertAll(normInputs, roundAtPlaceVal);
                    var outputControls = new[] {
                        a_rec, b_rec, c_rec,
                        d_rec, e_rec, ad_rec,
                        de_rec, ld_rec, fs_rec,
                        susp_rec, normal_rec,
                        suspect_rec, phato_rec };
                    Func<double, double> minmaxPercent = x => minmax(x, 1, -1) * 100;
                    var outputs = Classify(minmaxPercent, roundedNormInputs);
                    var clearedOutputControls = outputControls.Select(rec => rec.ResetWidth());
                    Func<Tuple<double, Rectangle>, Task<Rectangle>> progress = valueAndControl =>
                     ProgressAsync(ToInt16(valueAndControl.Item1), valueAndControl.Item2, EaseInOutCubic);
                    var progressed = outputs.Zip(clearedOutputControls, Tuple.Create).Select(progress);
                    var allTask = await WhenAll(progressed);
                    allTask.ToObservable().Take(1).Subscribe(_ => button.IsEnabled = true);
                });
        }

        double minToSec (double min) => min / 60;
        double minmax (double value, double max, double min = 0) => ( min.Equals(0) ) ? ( value / max ) : ( ( value - min ) / ( max - min ) );
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
    }
}