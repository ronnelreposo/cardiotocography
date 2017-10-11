
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

///The Dot Product of two Vectors.
let dotProduct xs ys =
 match xs, ys with
 | a::b, c::d -> vecOp (*) xs ys |> List.sum
 | _, _ -> 0.0

let square x = x * x

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

let rec dataAtIndex  xs_data xs_index =
 match xs_index with
 | [] -> []
 | hd::tl -> (List.item hd xs_data)::(dataAtIndex xs_data tl)

let rec f1 mapper x ys = List.map (mapper x) ys
let rec f2 mapper xs ys = //***normalize
 match xs with
 | [] -> []
 | hd::tl -> (f1 mapper hd ys)::(f2 mapper tl ys)

let sclarVectorMul x ys = f1 (*) x ys
let mulVectors xs ys    = List.map2 (*) xs ys
let addVectors xs ys    = List.map2 (+) xs ys
let logSigmoid x        = (/) 1.0 ((+) 1.0 (exp -x))
let dervLogSigmoid x    = (*) x ((-) 1.0 x)
let dervTanH x          = (*) (1.0 - x) (1.0 + x)
let gradient dFunc output target = (*) (dFunc output) ((-) target output)

let weightedSum inputs weights bias =
 addVectors bias (List.map (dotProduct inputs) weights)

let deltas N gradients net_outputs =
 List.map (sclarVectorMul N) (f2 (*) gradients net_outputs)

type Network = {
 N:float
 M:float
 Inputs:List<float>
 InputToHiddenWeights:List<List<float>>
 HiddenBias:List<float>
 HiddenPrevDeltas:List<List<float>>
 HiddenBiasPrevDeltas:List<float>
 HiddenNetOutputs:List<float>
 HiddenToOutputWeights:List<List<float>>
 OutputBias:List<float>
 OutputPrevDeltas:List<List<float>>
 OutputBiasPrevDeltas:List<float>
 Outputs:List<float>
 TargetOutputs:List<float>
 }

let feedForward net =
 let hiddenWeightedSum = weightedSum net.Inputs net.InputToHiddenWeights net.HiddenBias
 let hiddenNetOutputs = List.map tanh hiddenWeightedSum
 let outputWeightedSum = weightedSum hiddenNetOutputs net.HiddenToOutputWeights net.OutputBias
 let outputs = List.map tanh outputWeightedSum
 {
  net with
   HiddenNetOutputs = hiddenNetOutputs
   Outputs = outputs
 }

let backPropagate net =

 let out_grads = List.map2 (gradient dervTanH) net.Outputs net.TargetOutputs
 let out_deltas = deltas net.N out_grads net.HiddenNetOutputs
 let out_prevDeltasWithM = List.map (sclarVectorMul net.M) net.OutputPrevDeltas
 let out_newDeltas = List.map2 addVectors out_deltas out_prevDeltasWithM
 let out_hidden_weights_update= List.map2 addVectors net.HiddenToOutputWeights out_newDeltas
 let out_bias_deltas = sclarVectorMul net.N out_grads
 let out_bias_prevDeltasWithM = sclarVectorMul net.M net.OutputBiasPrevDeltas
 let out_bias_newDeltas = addVectors out_bias_deltas out_bias_prevDeltasWithM
 let out_bias_update = addVectors net.OutputBias out_bias_newDeltas
 
 let hid_grads = mulVectors (List.map dervTanH net.HiddenNetOutputs) (List.map (dotProduct out_grads) (transpose net.HiddenToOutputWeights))
 let hid_deltas = deltas net.N hid_grads net.Inputs
 let hid_prevDeltasWithM = List.map (sclarVectorMul net.M) net.HiddenPrevDeltas
 let hid_newDeltas = List.map2 addVectors hid_deltas hid_prevDeltasWithM
 let hid_input_weights_update = List.map2 addVectors net.InputToHiddenWeights hid_newDeltas
 let hid_bias_deltas = sclarVectorMul net.N hid_grads
 let hid_bias_prevDeltasWithM = sclarVectorMul net.M net.HiddenBiasPrevDeltas
 let hid_bias_newDeltas = addVectors hid_bias_deltas hid_bias_prevDeltasWithM
 let hid_bias_update = addVectors net.HiddenBias hid_bias_newDeltas

 {
  net with
   InputToHiddenWeights = hid_input_weights_update
   HiddenToOutputWeights = out_hidden_weights_update
   HiddenBias = hid_bias_update
   OutputBias = out_bias_update
   HiddenPrevDeltas = hid_deltas
   OutputPrevDeltas = out_deltas
   HiddenBiasPrevDeltas = hid_bias_deltas
   OutputBiasPrevDeltas = out_bias_deltas
 }

let rec train
 epoch
 netAcc
 ((training_samples:List<List<float>>), (teaching_inputs:List<List<float>>))
 ((testing_samples:List<List<float>>), (teaching_testing_inputs:List<List<float>>))
 =
 let trainOnce = feedForward >> backPropagate
 let rms network =
  (List.map2 (-) network.TargetOutputs network.Outputs)
  |> List.map square
  |> List.average
  |> sqrt
  |> (*) 0.5

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
  let rms_trained_err = rms trained

  let validated = List.fold2 (funcNet feedForward) netAcc shuffled_testing_s shuffled_testing_i
  let rms_validated_err = rms validated

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

let rec genWeights (rand:System.Random) count =
 match count with
 | 0 -> []
 | _ -> rand.NextDouble()::(genWeights rand ((-) count 1))
let genRandWeights = genWeights (System.Random())

let network = {
 N = 0.001
 M = 0.9
 Inputs = List.replicate inputSize 0.0
 InputToHiddenWeights = List.chunkBySize inputSize (genRandWeights ((*) inputSize hiddenSize))
 HiddenBias = genRandWeights hiddenSize
 HiddenToOutputWeights = List.chunkBySize hiddenSize (genRandWeights ((*) hiddenSize outputSize))
 HiddenNetOutputs = List.empty
 HiddenPrevDeltas = List.replicate hiddenSize (List.replicate inputSize 0.0)
 HiddenBiasPrevDeltas = List.replicate hiddenSize 0.0
 OutputBias = genRandWeights outputSize
 OutputPrevDeltas = List.replicate outputSize (List.replicate hiddenSize 0.0)
 OutputBiasPrevDeltas = List.replicate outputSize 0.0
 Outputs = List.empty
 TargetOutputs = List.replicate outputSize 0.0
}

let training_samples = [| [ 0.500; 0.000; 0.000; 0.000; 0.000; 0.000; 0.000;  ] |]
let teaching_inputs = [| [ 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 1.0; 0.0;  ] |]
let testing_samples = [| [ 0.692; 0.000; 0.000; 0.174; 0.000; 0.000; 0.000; ] |]
let teaching_samples_inputs = [| [ 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0;  ] |]

let epoch = 10000

let trained =
 train
  epoch network
  ((Array.toList training_samples), (Array.toList teaching_inputs))
  ((Array.toList testing_samples), (Array.toList teaching_samples_inputs))
