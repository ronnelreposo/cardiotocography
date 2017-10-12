
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
 HiddenLayer : Layer
 OutputLayer : Layer
 TargetOutputs: List<float>
 }

let feedForward net =
 let hiddenWeightedSum = weightedSum net.Inputs net.HiddenLayer.Weights net.HiddenLayer.Bias
 let hiddenNetOutputs = List.map tanh hiddenWeightedSum
 let outputWeightedSum = weightedSum hiddenNetOutputs net.OutputLayer.Weights net.OutputLayer.Bias
 let outputs = List.map tanh outputWeightedSum
 {
  net with
   HiddenLayer = { net.HiddenLayer with NetOutputs = hiddenNetOutputs }
   OutputLayer = { net.OutputLayer with NetOutputs = outputs }
 }

let backPropagate (net:Network) =

 let out_grads = List.map2 (gradient deltaTanH) net.OutputLayer.NetOutputs net.TargetOutputs
 let out_deltas = deltas net.LearningRate out_grads net.HiddenLayer.NetOutputs
 let out_prevDeltasWithM = List.map (smul net.Momentum) net.OutputLayer.PrevDeltas
 let out_newDeltas = List.map2 add out_deltas out_prevDeltasWithM
 let out_hidden_weights_update= List.map2 add net.OutputLayer.Weights out_newDeltas
 let out_bias_deltas = smul net.LearningRate out_grads
 let out_bias_prevDeltasWithM = smul net.Momentum net.OutputLayer.BiasPrevDeltas
 let out_bias_newDeltas = add out_bias_deltas out_bias_prevDeltasWithM
 let out_bias_update = add net.OutputLayer.Bias out_bias_newDeltas
 
 let hid_grads = mul (List.map deltaTanH net.HiddenLayer.NetOutputs) (List.map (dot out_grads) (transpose net.OutputLayer.Weights))
 let hid_deltas = deltas net.LearningRate hid_grads net.Inputs
 let hid_prevDeltasWithM = List.map (smul net.Momentum) net.HiddenLayer.PrevDeltas
 let hid_newDeltas = List.map2 add hid_deltas hid_prevDeltasWithM
 let hid_input_weights_update = List.map2 add net.HiddenLayer.Weights hid_newDeltas
 let hid_bias_deltas = smul net.LearningRate hid_grads
 let hid_bias_prevDeltasWithM = smul net.Momentum net.HiddenLayer.BiasPrevDeltas
 let hid_bias_newDeltas = add hid_bias_deltas hid_bias_prevDeltasWithM
 let hid_bias_update = add net.HiddenLayer.Bias hid_bias_newDeltas

 {
  net with
   HiddenLayer = {
                  net.HiddenLayer with
                   Weights = hid_input_weights_update
                   Bias = hid_bias_update
                   PrevDeltas = hid_deltas
                   BiasPrevDeltas = hid_bias_deltas
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
 ((training_samples:List<List<float>>), (teaching_inputs:List<List<float>>))
 ((testing_samples:List<List<float>>), (teaching_testing_inputs:List<List<float>>))
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
  let shuffled_training_i = dataAtIndex teaching_inputs training_rand_index

  let testing_index = [0..(testing_samples.Length - 1)]
  let testing_rand_index = shuffleList testing_index.Length rand testing_index
  let shuffled_testing_s = dataAtIndex testing_samples testing_rand_index
  let shuffled_testing_i = dataAtIndex teaching_testing_inputs testing_rand_index

  let trained = List.fold2 (funcNet trainOnce) netAcc shuffled_training_s shuffled_training_i
  let rms_trained_err = networkDistance trained

  let validated = List.fold2 (funcNet feedForward) netAcc shuffled_testing_s shuffled_testing_i
  let rms_validated_err = networkDistance validated

//  if epoch % 100 = 0 then printfn "%f %f" rms_trained_err rms_validated_err
  printfn "%f %f" rms_trained_err rms_validated_err
  
  let best_trained_err = 0.019999 //*** CONSIDER PASSING TO PARAMETER.
  let best_test_err = 0.019999
  let errTresholdMeet = (rms_trained_err < best_trained_err) && (rms_validated_err < best_test_err)
  if errTresholdMeet then trained
  else train ((-) epoch 1) trained (training_samples, teaching_inputs) (testing_samples, teaching_testing_inputs)

let inputSize = 7;
let hiddenSize = 10;
let outputSize = 13;

let network = {
 LearningRate = 0.001
 Momentum = 0.9
 Inputs = List.replicate inputSize 0.0
 HiddenLayer = {
                Inputs = List.empty
                Weights = ((*) inputSize hiddenSize) |> listRandElems |> List.chunkBySize inputSize
                Bias = listRandElems hiddenSize
                PrevDeltas = List.replicate hiddenSize <| List.replicate inputSize 0.0
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

let training_samples = [|
 [ 0.500; 0.000; 0.000; 0.000; 0.000; 0.000; 0.000;  ]
 [ 0.500; 0.000; 0.000; 0.000; 0.000; 0.000; 0.000;  ]
 |]
let teaching_inputs = [|
 [ 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 1.0; 0.0;  ]
 [ 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 1.0; 0.0;  ]
 |]
let testing_samples = [|
 [ 0.692; 0.000; 0.000; 0.174; 0.000; 0.000; 0.000; ]
 [ 0.692; 0.000; 0.000; 0.174; 0.000; 0.000; 0.000; ]
 |]
let teaching_samples_inputs = [|
 [ 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0;  ]
 [ 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0;  ]
 |]

let epoch = 20

let trained =
 train
  epoch network
  ((Array.toList training_samples), (Array.toList teaching_inputs))
  ((Array.toList testing_samples), (Array.toList teaching_samples_inputs))