
open System.IO;
open System.Text.RegularExpressions;

/// Matrix Transpose.
let transpose xss =
 let rec f xss acc =
  match xss with
  | [] -> failwith "xss must not contain empty vectors."
  | hd::_ ->
   match hd with
   | [] -> List.rev acc
   | _ ->
    f <| List.map (List.skip 1) xss <| (List.map List.head xss)::acc
 f xss List.empty

///Operation of two Vectors.
let vecOp f xs ys =
 match xs, ys with
 | [], [] -> []
 | [], hd::tl -> ys
 | a::b, [] -> xs
 | a::b, c::d -> List.map2 f xs ys

/// The Dot Product of xs and ys.
let dot xs ys = List.map2 (*) xs ys |> List.sum

let square x = x * x

let distance xs ys =
  (List.map2 (-) xs ys)
  |> List.map square
  |> List.average
  |> sqrt
  |> (*) 0.5

/// Euclid Norm.
let rec norm xs =
 let rec f xs acc =
  match xs with
  | [] -> List.sum acc |> sqrt
  | hd::tl -> f tl ((square hd)::acc)
 f xs List.empty

///Naive Shuffle List.
let rec shuffleList count (sysRand:System.Random) (xs:List<'a>) =
 match count with
 | 0 -> xs
 | _ ->
  let random = sysRand.Next(((-) count 1), xs.Length)
  let item = List.item random xs
  let filtered = List.filter (fun a -> a <> item) xs
  let b, c = List.splitAt random filtered
  let randomxs = List.append (item::b) (List.rev c)
  shuffleList ((-) count 1) sysRand randomxs

/// Retrieves the value of data on a list given the list of index.
let dataAtIndex  xs_data xs_index =
 let rec f data_xs index_xs acc =
  match index_xs with
  | [] -> List.rev acc
  | hd::tl -> (f data_xs tl ((List.item hd data_xs)::acc))
 f xs_data xs_index List.empty

/// Maps a scalar to a vector using a mapper function.
let scalarToVecOp mapper x ys = List.map (mapper x) ys

/// Maps the elements the first list (xs) to second list (ys) using the mapper function.
/// first, it gets the first element of first list (xs) and maps to second list (ys)
/// using the mapper function. i.e. (List.map (mapper x) ys).
/// Finally it returns the accumulated mapped list.
/// mapToSecondList (+) ["1"; "2"; "3"] ["2"; "3", "4"] =
/// [ ["12"; "13"; "14"]; ["22"; "23"; "24"]; ["32"; "33"; "34"] ].
let mapToSecondList mapper xs ys =
 let rec f mapper xs ys acc = match xs with | [] -> List.rev acc | hd::tl -> f mapper tl ys <| (List.map (mapper hd) ys)::acc
 f mapper xs ys List.empty

/// Scalar Vector Multiplication.
let smul c xs = List.map ((*) c) xs

/// Vector Multiplication.
let mul xs ys = List.map2 (*) xs ys

/// Vector Addition.
let add xs ys = List.map2 (+) xs ys

/// Logistic Sigmoid.
let logSigmoid x = (/) 1.0 ((+) 1.0 (exp -x))

/// Derivative of Logistic Sigmoid.
let deltaLogSigmoid x = (*) x ((-) 1.0 x)

/// Derivative of TanH i.e. sec^2h.
let deltaTanH x = (/) 1.0 <| (*) (cosh x) (cosh x)

/// Generate List of Random Elements.
let listRandElems count =
 let rec f (rand:System.Random) acc c = match c with | 0 -> acc | _ -> f rand <| rand.NextDouble()::acc <| (-) c 1
 f (System.Random()) List.empty count

/// Gradient. dFunc is the derivative of forward squashing function.
let gradient dFunc output target = (*) <| dFunc output <| (-) target output

/// Weighted Sum with Bias.
let weightedSum inputs weights bias = add bias <| List.map (dot inputs) weights

/// Delta or The Rate of Change.
let deltas learningRate gradients netOutputs = List.map <| smul learningRate <| mapToSecondList (*) gradients netOutputs

/// Represents a Network Layer.
type Layer = {
  Inputs: List<float>
  Weights: List<List<float>>
  Bias: List<float>
  PrevDeltas: List<List<float>>
  BiasPrevDeltas: List<float>
  NetOutputs: List<float>
  }

/// Represents a Feed Forward Network.
type Network = {
 LearningRate: float
 Momentum: float
 Inputs: List<float>
 FirstHiddenLayer : Layer
 SecondHiddenLayer : Layer
 OutputLayer : Layer
 TargetOutputs: List<float>
 }

let feedForward net =

 let firstHiddenWeightedSum = weightedSum net.Inputs net.FirstHiddenLayer.Weights net.FirstHiddenLayer.Bias
 let firstHiddenNetOutputs = List.map tanh firstHiddenWeightedSum
 let secondHiddenWeightedSum = weightedSum firstHiddenNetOutputs net.SecondHiddenLayer.Weights net.SecondHiddenLayer.Bias
 let secondHiddenNetOutputs = List.map tanh secondHiddenWeightedSum

 let outputWeightedSum = weightedSum secondHiddenNetOutputs net.OutputLayer.Weights net.OutputLayer.Bias
 let outputs = List.map tanh outputWeightedSum
 {
  net with
   FirstHiddenLayer = { net.FirstHiddenLayer with NetOutputs = firstHiddenNetOutputs }
   SecondHiddenLayer = { net.SecondHiddenLayer with NetOutputs = secondHiddenNetOutputs }
   OutputLayer = { net.OutputLayer with NetOutputs = outputs }
 }

(* *** note: the previous implementation, newDeltas are used instead of prevDeltas. *)
let backPropagate (net:Network) = (* OutputLayer->SecondHiddenLayer->FirstHiddenLayer->Inputs *)

 let out_grads = List.map2 (gradient deltaTanH) net.OutputLayer.NetOutputs net.TargetOutputs
 let out_deltas = deltas net.LearningRate out_grads net.SecondHiddenLayer.NetOutputs
 let out_prevDeltasWithM = List.map (smul net.Momentum) net.OutputLayer.PrevDeltas
 let out_newDeltas = List.map2 add out_deltas out_prevDeltasWithM
 let out_hidden_weights_update= List.map2 add net.OutputLayer.Weights out_newDeltas
 let out_bias_deltas = smul net.LearningRate out_grads
 let out_bias_prevDeltasWithM = smul net.Momentum net.OutputLayer.BiasPrevDeltas
 let out_bias_newDeltas = add out_bias_deltas out_bias_prevDeltasWithM
 let out_bias_update = add net.OutputLayer.Bias out_bias_newDeltas

 let secHidGrads = mul (List.map deltaTanH net.SecondHiddenLayer.NetOutputs) (List.map (dot out_grads) (transpose net.OutputLayer.Weights))
 let secHidDeltas = deltas net.LearningRate secHidGrads net.FirstHiddenLayer.NetOutputs
 let secHidPrevDeltasWithM = List.map (smul net.Momentum) net.SecondHiddenLayer.PrevDeltas
 let secHidNewDeltas = List.map2 add secHidDeltas secHidPrevDeltasWithM
 let secHidInputWeightsUpdate = List.map2 add net.SecondHiddenLayer.Weights secHidNewDeltas
 let secHidBiasDeltas = smul net.LearningRate secHidGrads
 let secHidBiasPrevDeltasWithM = smul net.Momentum net.SecondHiddenLayer.BiasPrevDeltas
 let secHidBiasNewDeltas = add secHidBiasDeltas secHidBiasPrevDeltasWithM
 let secHidBiasUpdate = add net.SecondHiddenLayer.Bias secHidBiasNewDeltas

 let firstHidGrads = mul (List.map deltaTanH net.FirstHiddenLayer.NetOutputs) (List.map (dot secHidGrads) (transpose net.SecondHiddenLayer.Weights))
 let firstHidDeltas = deltas net.LearningRate firstHidGrads net.Inputs
 let firstHidPrevDeltasWithM = List.map (smul net.Momentum) net.FirstHiddenLayer.PrevDeltas
 let firstHidNewDeltas = List.map2 add firstHidDeltas firstHidPrevDeltasWithM
 let firstHidInputWeightsUpdate = List.map2 add net.FirstHiddenLayer.Weights firstHidNewDeltas
 let firstHidBiasDeltas = smul net.LearningRate firstHidGrads
 let firstHidBiasPrevDeltasWithM = smul net.Momentum net.FirstHiddenLayer.BiasPrevDeltas
 let firstHidBiasNewDeltas = add firstHidBiasDeltas firstHidBiasPrevDeltasWithM
 let firstHidBiasUpdate = add net.FirstHiddenLayer.Bias firstHidBiasNewDeltas
 {
  net with
   FirstHiddenLayer = {
                       net.FirstHiddenLayer with
                        Weights = firstHidInputWeightsUpdate
                        Bias = firstHidBiasUpdate
                        PrevDeltas = firstHidNewDeltas
                        BiasPrevDeltas = firstHidBiasNewDeltas
                   }
   SecondHiddenLayer = {
                        net.SecondHiddenLayer with
                         Weights = secHidInputWeightsUpdate
                         Bias = secHidBiasUpdate
                         PrevDeltas = secHidNewDeltas
                         BiasPrevDeltas = secHidBiasNewDeltas
                   }
   OutputLayer = {
                  net.OutputLayer with
                   Weights = out_hidden_weights_update
                   Bias = out_bias_update
                   PrevDeltas = out_deltas
                   BiasPrevDeltas = out_bias_deltas
                   }
 }

let rec train
 epoch
 netAcc
 ((training_samples:List<List<float>>), (teachingInputs:List<List<float>>))
 ((testing_samples:List<List<float>>), (testOutputs:List<List<float>>))
 =
 let trainOnce = feedForward >> backPropagate

 let networkDistance network = distance network.TargetOutputs network.OutputLayer.NetOutputs

 match epoch with
 | 0 -> netAcc
 | _ ->

  let funcNet func net inputs targets = func { net with Inputs = inputs; TargetOutputs = targets }

  let rand = new System.Random()

  let training_index = [0..(training_samples.Length - 1)]
  let training_rand_index = shuffleList training_index.Length rand training_index
  let shuffled_training_s = dataAtIndex training_samples training_rand_index
  let shuffled_training_i = dataAtIndex teachingInputs training_rand_index

  let testing_index = [0..(testing_samples.Length - 1)]
  let testing_rand_index = shuffleList testing_index.Length rand testing_index
  let shuffled_testing_s = dataAtIndex testing_samples testing_rand_index
  let shuffled_testing_i = dataAtIndex testOutputs testing_rand_index

  let trained = List.fold2 (funcNet trainOnce) netAcc shuffled_training_s shuffled_training_i
  let rms_trained_err = networkDistance trained

  let validated = List.fold2 (funcNet feedForward) netAcc shuffled_testing_s shuffled_testing_i
  let rms_validated_err = networkDistance validated

  if epoch % 100 = 0 then printfn "%f %f" rms_trained_err rms_validated_err
//  printfn "%f %f" rms_trained_err rms_validated_err
  
//  let best_trained_err = 0.019999 //*** CONSIDER PASSING TO PARAMETER.
//  let best_test_err = 0.019999
//  let errTresholdMeet = (rms_trained_err < best_trained_err) && (rms_validated_err < best_test_err)
//  if errTresholdMeet then trained
//  else train ((-) epoch 1) trained (training_samples, teachingInputs) (testing_samples, testOutputs)
  train ((-) epoch 1) trained (training_samples, teachingInputs) (testing_samples, testOutputs)

let inputSize = 7;
let hiddenSize = 10;
let outputSize = 13;

let network = {
 LearningRate = 0.001
 Momentum = 0.9
 Inputs = List.replicate inputSize 0.0
 FirstHiddenLayer = {
                     Inputs = List.empty
                     Weights = ((*) inputSize hiddenSize) |> listRandElems |> List.chunkBySize inputSize
                     Bias = listRandElems hiddenSize
                     PrevDeltas = List.replicate hiddenSize <| List.replicate inputSize 0.0
                     BiasPrevDeltas = List.replicate hiddenSize 0.0
                     NetOutputs = List.empty
 }
 SecondHiddenLayer = {
                      Inputs = List.empty
                      Weights = ((*) hiddenSize hiddenSize) |> listRandElems |> List.chunkBySize hiddenSize
                      Bias = listRandElems hiddenSize
                      PrevDeltas = List.replicate hiddenSize <| List.replicate hiddenSize 0.0
                      BiasPrevDeltas = List.replicate hiddenSize 0.0
                      NetOutputs = List.empty
 }
 OutputLayer = {
                Inputs = List.empty
                Weights = ((*) hiddenSize outputSize) |> listRandElems |> List.chunkBySize hiddenSize
                Bias = listRandElems outputSize
                PrevDeltas = List.replicate outputSize <| List.replicate hiddenSize 0.0
                BiasPrevDeltas = List.replicate outputSize 0.0
                NetOutputs = List.empty
 }
 TargetOutputs = List.replicate outputSize 0.0
}

let dataToFloatList separator data = Regex.Split(data, separator) |> Array.Parallel.map float |> Array.toList
let csvStrToFloatList = dataToFloatList ","
let data filename = (* replace with your current directory. *)
 File.ReadAllLines(@"D:\Projects\AI\cardiotocography\data and error\"+filename)
 |> Array.toList
 |> List.map csvStrToFloatList

let training_samples = data "training_samples.txt"
let teaching_inputs = data "teaching_inputs.txt"
let testing_samples = data "testing_samples.txt"
let test_outputs = data "test_outputs.txt"

let epoch = 1

let trained = train epoch network (training_samples, teaching_inputs) (testing_samples, test_outputs)

(* val trained : Network = //Error: (Training: 0.016551, Test: 0.017999)
  {LearningRate = 0.001;
   Momentum = 0.9;
   Inputs = [0.508; 0.192; 0.005; 0.13; 0.375; 0.0; 0.0];
   FirstHiddenLayer =
    {Inputs = [];
     Weights =
      [[1.726281808; -3.654357537; 0.537142092; -7.07135477; -2.256975369; -0.7940866435; -1.213602292];
       [0.9145099682; -0.7068201194; -0.935228927; 1.664930999; 1.125694609; 0.761819344; 1.013573637];
       [-0.6117482449; -0.837369781; 0.2285114287; -0.8287770466; 6.392127843; 0.1256154636; 2.92461257];
       [-3.534252779; -5.290860967; 0.360950828; -2.508273421; 1.904448799; 0.671905326; 3.654652122];
       [5.299449096; -5.062333147; 1.427488028; 2.500663245; 1.878716237; 1.017709897; 2.075505693];
       [-2.669645158; 8.402094978; 0.4208968611; 0.7517770645; 0.9030522328; 0.7055006023; 1.157783545];
       [0.190001696; 4.527113937; 1.202078967; -1.614895953; 0.2476088163; -0.6216961795; -0.505429305];
       [0.698495376; 6.146216178; -0.1615875242; 0.08715846116; 3.173732066; 0.8996711787; 1.025651132];
       [-0.3593885325; 2.032299941; 3.844176294; -9.176315707; -4.871621287; -0.9928479914; 3.951703723];
       [-1.02824285; 6.071786973; 0.6703246096; 0.8240747304; 2.490067237; 1.028164132; 1.526152574]];
     Bias =
      [-1.174995304; -0.5308293614; -0.1459036789; 1.370614004; -2.986152089; 1.302194378; -0.9339367268; -0.4650364059; 0.4175758211; -0.6261257587];
     PrevDeltas =
      [[-0.0007018613477; -7.983390594e-05; -2.518690365e-05; -7.064189857e-05; -9.442396732e-06; -2.48982465e-22; -5.239618287e-05];
       [-0.0001058298544; 2.466751155e-05; -8.392312062e-06; 1.742023972e-05; -2.79269585e-05; 4.401674124e-22; 2.224892223e-05];
       [-0.0001411680682; -8.029169845e-05; -2.626076778e-05; -9.597039045e-05; -0.0001022678273; 1.68476738e-21; 0.0001050732444];
       [-0.0006323227219; -5.496907946e-05; -1.859198162e-05; -3.326111338e-05; -1.080009663e-07; 4.993571281e-22; -4.619002364e-05];
       [-0.0003290565499; 2.397335069e-05; 1.438814515e-05; 1.762374405e-05; -1.123613183e-05; 1.512452695e-21; -3.846234113e-05];
       [0.001015764651; 0.0001206640743; 3.23542858e-05; 4.156515073e-05; 3.445482309e-05; -1.76105588e-21; 2.52282842e-05];
       [0.0003275529785; -2.960005329e-05; -2.650618403e-06; -4.533192108e-05; 1.039038701e-05; -1.144376404e-21; 2.959496501e-06];
       [0.0003625077399; 3.721624808e-05; 6.18343583e-05; -5.78516154e-05; 1.057500968e-05; 1.097461144e-21; 3.922897819e-05];
       [-0.001055081547; -0.0001707704719; -5.072159583e-05; -9.392788646e-05; -2.141116148e-05; -1.175196255e-21; -3.779967674e-05];
       [-0.0002216452497; -1.401030813e-05; -6.194608862e-05; 8.950003261e-05; 9.007235338e-07; -1.837478025e-21; -9.44828331e-06]];
     BiasPrevDeltas =
      [-0.00110869477; -0.0001337555981; -0.0003052296352; -0.0009526107834; -0.0004190446203; 0.001432757479; 0.0003729308499; 0.0004385942466; -0.001631412522; -0.0002955529144];
     NetOutputs =
      [-0.9920384267; 0.4070945319; 0.9320058274; -0.7815906325; -0.2247699178; 0.963722985; -0.07964534604; 0.9788752085; -0.9827812835; 0.7864150348];};
   SecondHiddenLayer =
    {Inputs = [];
     Weights =
      [[3.593616455; 0.6573932997; 4.773306211; 3.421190959; 4.741637129; -5.400577376; -2.868604026; 1.207042627; 0.1680596223; -4.883681135];
       [5.988640048; -2.451498621; 0.006509786615; 5.146810636; -5.465599992; -2.696425868; 3.21534413; -3.676445104; 9.708397944; 0.8725269354];
       [0.006564210884; 0.1350889479; 0.3997738036; 0.3923544062; -0.059506501; -0.4016720287; -0.003846411874; -0.5531319045; -0.1599027475; -0.1878434774];
       [-1.117998343; 0.12756794; 1.757414892; -1.187957115; -0.03379757443; -0.4294342182; 0.3456648526; -1.927628054; -0.4422991529; -0.0464167116];
       [-0.08250938446; 0.2090950114; 0.1709865896; 0.7286006938; 0.2256582578; -0.5219112359; -0.07081632068; 0.6775586385; 0.4031370587; 0.7609691937];
       [0.4115042181; -0.3977266509; -0.8881097521; -0.4002101428; -0.1437509611; 0.6236012139; -0.1394600089; -0.004568496093; 0.239837531; 0.1506719304];
       [0.3114711516; 0.665294265; -1.970315482; 4.068437999; -0.3271810699; 1.544609945; -0.1492158931; -1.21682281; 0.4766890969; 1.880151042];
       [0.1504099641; -0.7234894885; -2.140861549; -0.07104646203; 0.08815223069; 1.190553954; 0.3982424584; 1.035756909; -0.006999053232; -0.3326504712];
       [1.492333244; 0.3065644182; -1.662657036; -0.2700319477; 3.123000689; -0.8969118629; -0.8124379536; -1.782670912; 1.521321442; -0.03831239805];
       [0.615112193; 0.1494468573; -0.4795026101; -0.7849598083; -0.04818084373; 2.283872996; 1.150335914; 1.578760203; -0.1586200703; -0.8218845669]];
     Bias =
      [-0.699734971; -3.132900567; 1.74920407; -0.5229081871; 0.9983439443; 0.2654260609; -0.6495168154; -1.85842874; -1.224193; -1.70780481];
     PrevDeltas =
      [[3.862115645e-05; -2.489484875e-05; 2.774796991e-05; 6.503279334e-05; -1.301465975e-05; -2.513213903e-05; 3.531156352e-05; -3.1770828e-05; 3.596046637e-05; 1.553572473e-06];
       [6.435289646e-05; -1.085230528e-05; 6.456620125e-05; 9.418319988e-05; 4.37128655e-05; -6.276077986e-05; 1.78239792e-05; -8.207328553e-05; 8.676948683e-07; -3.005335583e-05];
       [-0.0001549578328; -3.736975108e-05; -0.0001544409594; -0.0002782511078; 4.115553868e-05; 0.0001055665155; -6.033141242e-05; 0.0001913804264; 3.133938782e-06; -0.0001909287197];
       [-1.329748597e-05; 8.001023056e-05; -0.0001559502603; -0.0002647367913; 0.0002396344414; -0.0001071005183; -0.0002931362238; 2.534256358e-05; 8.914994074e-05; -0.0002763643536];
       [0.0001858718458; -1.024071252e-06; 8.046012203e-05; 0.0001895388093; 8.957771641e-05; -0.0001676625218; -5.667852985e-05; -0.0002271023461; 0.0001138800013; -7.472758535e-05];
       [-6.702084652e-06; 8.854407482e-05; -5.292145633e-05; -0.0001948753282; 0.0001885939682; -5.369655494e-05; -0.0002551603998; 6.843441516e-06; 5.281317699e-05; -0.0001711794998];
       [-5.826407158e-05; 1.788795396e-05; 2.831516479e-05; -9.852859688e-06; -7.146356026e-05; 0.0001234302682; 3.622067039e-05; 4.410887363e-05; -8.722656379e-05; 0.0001210640244];
       [-0.000152710038; -6.452138263e-06; -6.094470651e-05; -0.0002980051305; 6.413770242e-05; 0.0001298385089; -0.000117942243; 0.0002244027831; -8.991777195e-05; -0.0001368357949];
       [6.741709088e-05; -3.545559653e-05; 6.23856241e-05; 0.0001263324496; -0.0001612057822; 1.516797613e-05; 0.0001310445857; -0.0001115760603; -9.166478865e-05; 0.0001520380804];
       [-2.233165472e-05; 7.936451093e-06; -0.0001761239513; -0.0001973518581; 0.0002300641152; -0.0001732705525; -0.0001267098346; 6.198574195e-05; 0.0001089210811; -0.0003224513595]];
     BiasPrevDeltas =
      [-6.495844869e-05; -0.0001263384782; 0.0003388882679; 0.0003811863256;
       -0.0001806551684; 0.000296230462; -3.336944693e-05; 0.0003730293965;
       -0.0002607504523; 0.000266807499];
     NetOutputs =
      [-0.9999999995; -1.0; 0.7431575028; 0.8570029145; 0.7908317491;
       -0.2886726294; -0.9996400886; -0.9832805851; -0.9999999434;
       0.7949910384];};
   OutputLayer =
    {Inputs = [];
     Weights =
      [[-0.2247283172; 0.09174386932; 0.02918685363; 0.3955840698;
        0.08049351286; 0.402816426; 0.2566540433; 0.1878389645; 0.0310532791;
        -0.4725924912];
       [0.09407953105; 0.1883419936; 1.110169594; -0.2969248134; -0.4182465048;
        -0.4770819635; -0.1625491799; 0.7724049868; -0.016106866; 0.4190237911];
       [-0.01383037811; 0.2960953414; 0.3706114766; 0.1609421046; 0.2482632579;
        0.306240307; 0.08355896127; 0.114103727; -0.03025850249; -0.1010267164];
       [-0.002161124466; 0.7726468425; 0.07767010181; 0.2509410599;
        0.3811518787; 0.3282886052; 0.1506891106; 0.2677240596; 0.01068974533;
        -0.1355277339];
       [-0.01205407567; 0.3178001276; 0.5471316748; 0.2657302795; 0.3071523247;
        0.2795036379; 0.1677669581; 0.2821240187; 0.09026716979;
        -0.04281364267];
       [-0.1679868464; 0.7301158052; 0.2067576898; 0.309905898; 0.4624099716;
        0.379333787; 0.1886810123; -0.7552076401; 0.1637129867; 0.6482509813];
       [0.3152539732; 0.3074824395; 0.4272404012; 0.09937055852; 0.3568472204;
        -0.001252676324; -0.4842298395; 0.2965343753; -0.4253043711;
        -0.2044746814];
       [0.003185421846; 0.0147879532; 0.4595941053; 0.1084169372; 0.1986785644;
        0.1369402444; 0.9421718609; 0.1515499022; 0.002395559057;
        -0.01056331462];
       [0.01229553541; 0.9949111717; 0.5707429817; 0.1226082669; 0.3723834505;
        0.376929203; 0.05817434644; 0.1120186156; 0.01357500931;
        -0.05050360821];
       [0.03405416427; 0.3395262068; 0.30983338; 0.08572709147; 0.2198022165;
        0.2420675832; 0.03795175732; 0.04983253452; 0.2001719787;
        -0.008598148362];
       [0.07790858126; 0.1011500192; 1.124900389; 0.7820351866; 0.4926669283;
        0.921107303; -0.0560566102; 0.6100140528; -0.48410874; 0.8506866083];
       [0.03509922079; 0.0309950889; 0.3869894233; 0.11159615; 0.2686646989;
        0.2029735843; 0.03119030324; 0.1366887931; 0.270467283; -0.03481506217];
       [0.02026104164; 0.04881579955; 0.6594226675; 0.09349374659;
        0.3466535185; 0.3233205209; 1.013976173; 0.1143656709; 0.01261207352;
        -0.02655530324]];
     Bias =
      [0.3791637524; 0.1708510745; 0.01074344215; 0.7961274689; 0.1055304186;
       0.5763772184; -0.458935533; 0.5881191716; 0.5037707702; 0.2698262498;
       0.7204214873; 0.02517360795; 0.4712613344];
     PrevDeltas =
      [[-2.786399151e-05; -2.786399152e-05; 2.070733436e-05; 2.387952194e-05;
        2.203572915e-05; -8.043571698e-06; -2.785396295e-05; -2.739812189e-05;
        -2.786398994e-05; 2.215162355e-05];
       [1.774370862e-05; 1.774370863e-05; -1.31863702e-05; -1.520641001e-05;
        -1.403228813e-05; 5.122123026e-06; 1.773732247e-05; 1.744704421e-05;
        1.774370763e-05; -1.410608935e-05];
       [4.032767207e-06; 4.032767209e-06; -2.996981208e-06; -3.456093252e-06;
        -3.189240346e-06; 1.164149514e-06; 4.03131577e-06; 3.965341701e-06;
        4.032766981e-06; -3.206013791e-06];
       [-2.742607191e-05; -2.742607193e-05; 2.038189112e-05; 2.350422358e-05;
        2.168940843e-05; -7.917156297e-06; -2.741620097e-05; -2.696752405e-05;
        -2.742607037e-05; 2.18034814e-05];
       [2.728540371e-05; 2.728540373e-05; -2.02773525e-05; -2.338367052e-05;
        -2.157816355e-05; 7.876549238e-06; 2.72755834e-05; 2.682920774e-05;
        2.728540218e-05; -2.169165144e-05];
       [-3.73920776e-05; -3.739207762e-05; 2.778820302e-05; 3.20451195e-05;
        2.957084214e-05; -1.079406936e-05; -3.737861978e-05; -3.676690396e-05;
        -3.73920755e-05; 2.972636661e-05];
       [5.899040882e-05; 5.899040885e-05; -4.383916493e-05; -5.055495231e-05;
        -4.665148821e-05; 1.702891643e-05; 5.896917753e-05; 5.800412373e-05;
        5.899040551e-05; -4.689684639e-05];
       [2.073715017e-05; 2.073715018e-05; -1.541096875e-05; -1.777179815e-05;
        -1.639959675e-05; 5.98624767e-06; 2.072968665e-05; 2.039043717e-05;
        2.073714901e-05; -1.648584856e-05];
       [-1.048649491e-05; -1.048649492e-05; 7.793117375e-06; 8.986956706e-06;
        8.293053116e-06; -3.027164061e-06; -1.048272071e-05; -1.031116686e-05;
        -1.048649432e-05; 8.336669483e-06];
       [1.061689771e-05; 1.061689772e-05; -7.890027194e-06; -9.098712287e-06;
        -8.396179791e-06; 3.06480778e-06; 1.061307657e-05; 1.04393894e-05;
        1.061689712e-05; -8.440338541e-06];
       [-3.208498644e-06; -3.208498645e-06; 2.384419841e-06; 2.74969269e-06;
        2.537382596e-06; -9.262057404e-07; -3.20734387e-06; -3.154854426e-06;
        -3.208498464e-06; 2.55072767e-06];
       [3.337651015e-05; 3.337651017e-05; -2.480400395e-05; -2.860376649e-05;
        -2.639520391e-05; 9.63488495e-06; 3.336449758e-05; 3.281847445e-05;
        3.337650828e-05; -2.653402648e-05];
       [-6.502523274e-06; -6.502523277e-06; 4.83239896e-06; 5.572681401e-06;
        5.142401857e-06; -1.877100492e-06; -6.500182945e-06; -6.393804893e-06;
        -6.502522909e-06; 5.169447732e-06]];
     BiasPrevDeltas =
      [2.786399152e-05; -1.774370863e-05; -4.032767209e-06; 2.742607193e-05;
       -2.728540373e-05; 3.739207762e-05; -5.899040885e-05; -2.073715018e-05;
       1.048649492e-05; -1.061689772e-05; 3.208498645e-06; -3.337651017e-05;
       6.502523277e-06];
     NetOutputs =
      [-0.02788566447; 0.01774929915; 0.004032832797; -0.02744673782;
       0.02730575289; 0.9208603549; 0.05919737223; 0.02074607673;
       -0.01048764838; 0.01061809475; 0.9924474529; 0.03341378829;
       -0.00650279825];};
   TargetOutputs =
    [0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0];} *)