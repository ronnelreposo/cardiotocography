using FirstFloor.ModernUI.Windows.Controls;
using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Shapes;

namespace cardio
{
    public partial class MainWindow: ModernWindow
    {
        public MainWindow() { InitializeComponent(); }

        double minToSec(double min) => min / 60;
        double minmax(double value, double max) => value / max;
        double minmax(double value, double min, double max) => (value - min) / (max - min);
        double getDoubleValue(TextBox tb) => tb.Text == ""? 0 : Convert.ToDouble(tb.Text);

        async void classify_button_Click(object sender, RoutedEventArgs e)
        {
            var place_val = 3;

            var lb_convToSec = Math.Round(minToSec(getDoubleValue(lb_tb)), place_val);
            var lb_max = 5;
            var ac_max = 30;
            var fm_max = 600;
            var uc_max = 30;
            var dl_max = 20;

            classify_button.IsEnabled = false;

            var input_norm_xs = new double[] {
                minmax(lb_convToSec, 1, lb_max),
                minmax(getDoubleValue(ac_tb), ac_max),
                minmax(getDoubleValue(fm_tb), fm_max),
                minmax(getDoubleValue(uc_tb), uc_max),
                minmax(getDoubleValue(dl_tb), dl_max),
                getDoubleValue(ds_tb),
                minmax(getDoubleValue(dp_tb), lb_max) };

            var rounded_input_norm_xs = Array.ConvertAll(input_norm_xs, x => Math.Round(x, place_val));

            var output_controls_xs = new Rectangle[] { a_rec, b_rec, c_rec, d_rec, e_rec, ad_rec, de_rec, ld_rec, fs_rec, susp_rec, normal_rec, suspect_rec, phato_rec };
            var output_xs = Classifier.classify(x => x < 0 ? 0 : x * 100, rounded_input_norm_xs);
            var task_xs = new Task[output_xs.Length];

            Func<double, double, bool> condition = (a, b) => a == b;

            await Task.WhenAll(fmap(0, (value, rec) => progress(0, 1, condition, (a, b) => a - b, rec), task_xs, output_xs, output_controls_xs));

            await Task.WhenAll(fmap(0, (value, rec) => progress(Convert.ToInt16(value), 1, condition, (a, b) => a + b, rec), task_xs, output_xs, output_controls_xs));

            classify_button.IsEnabled = true;
        }

        Rectangle changeWidth(double width, Rectangle rec)
        {
            rec.Width = width;
            return rec;
        }

        async Task<Rectangle> progress(int max, int delay, Func<double, double, bool> cond, Func<double, double, double> delta, Rectangle rec)
        {
            if (cond(rec.Width, max)) { return rec; }

            await Task.Delay(delay);

            return await progress(max, delay, cond, delta, changeWidth(delta(rec.Width, 1), rec));
        }

        Task[] fmap(int i, Func<double, Rectangle, Task> mapper, Task[] acc, double[] d_xs, Rectangle[] pb_xs)
        {
            if (i > (acc.Length - 1)) { return acc; }
            acc[i] = mapper(d_xs[i], pb_xs[i]);
            return fmap((i + 1), mapper, acc, d_xs, pb_xs);
        }
    }
}