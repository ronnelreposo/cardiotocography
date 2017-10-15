
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

///Shuffle List (Fisher Yates Alogrithm).
let shuffle (rand: System.Random) (xs:List<'a>) =
 let rec shuffleTo (indexes: int[]) upTo =
  match upTo with
  | 0 -> indexes
  | _ ->
   let fst = rand.Next(upTo)
   let temp = indexes.[fst]
   indexes.[fst] <- indexes.[upTo]
   indexes.[upTo] <- temp
   shuffleTo indexes (upTo - 1)
 let length = xs.Length
 let indexes = [| 0 .. length - 1 |]
 let shuffled = shuffleTo indexes (length - 1)
 List.permute (fun i -> shuffled.[i]) xs

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

let vectorToString (vector:List<float>) =
 let concatCommaSep (x:float) s = x.ToString("F6") + "," + s
 List.foldBack concatCommaSep vector ""

let rec matrixToString (matrix:List<List<float>>) =
 let concatStringVector vector s = vectorToString vector + ";" + s
 List.foldBack concatStringVector matrix ""

/// Represents a Network Layer.
type Layer = {
  Inputs: List<float>
  Weights: List<List<float>>
  Bias: List<float>
  Gradients: List<float>
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
   FirstHiddenLayer = {
                       net.FirstHiddenLayer with
                        Inputs = net.Inputs
                        NetOutputs = firstHiddenNetOutputs
                      }
   SecondHiddenLayer = {
                        net.SecondHiddenLayer with
                         Inputs = firstHiddenNetOutputs
                         NetOutputs = secondHiddenNetOutputs
                       }
   OutputLayer = {
                  net.OutputLayer with
                   Inputs = secondHiddenNetOutputs
                   NetOutputs = outputs
                 }
 }

let bpOutputLayer n m tOutputs (layer:Layer) =
 let grads = List.map2 (gradient deltaTanH) layer.NetOutputs tOutputs
 let bpDeltas = deltas n grads layer.Inputs
 let prevDeltasWithM = List.map (smul m) layer.PrevDeltas
 let newDeltas = List.map2 add bpDeltas prevDeltasWithM
 let weightsUpdate= List.map2 add layer.Weights newDeltas
 let biasDeltas = smul n grads
 let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
 let biasNewDeltas = add biasDeltas biasPrevDeltasWithM
 let biasUpdate = add layer.Bias biasNewDeltas
 {
  layer with
   Weights = weightsUpdate
   Bias = biasUpdate
   Gradients = grads
   PrevDeltas = newDeltas
   BiasPrevDeltas = biasNewDeltas
 }

let bpHiddenLayer n m layer nextLayer =
 let grads = mul (List.map deltaTanH layer.NetOutputs) (List.map (dot nextLayer.Gradients) (transpose nextLayer.Weights))
 let bpDeltas = deltas n grads layer.Inputs
 let prevDeltasWithM = List.map (smul m) layer.PrevDeltas
 let newDeltas = List.map2 add bpDeltas prevDeltasWithM
 let weightsUpdate = List.map2 add layer.Weights newDeltas
 let biasDeltas = smul n grads
 let biasPrevDeltasWithM = smul m layer.BiasPrevDeltas
 let biasNewDeltas = add biasDeltas biasPrevDeltasWithM
 let biasUpdate = add layer.Bias biasNewDeltas
 {
  layer with
   Weights = weightsUpdate
   Bias = biasUpdate
   Gradients = grads
   PrevDeltas = newDeltas
   BiasPrevDeltas = biasNewDeltas
 }

let backPropagate (net:Network) =
 let bpOutputLayer = bpOutputLayer net.LearningRate net.Momentum net.TargetOutputs net.OutputLayer
 let bpHidLayerWithHyperParams = bpHiddenLayer net.LearningRate net.Momentum
 let bpSecHidLayer = bpHidLayerWithHyperParams net.SecondHiddenLayer bpOutputLayer
 let bpFirstHidLayer = bpHidLayerWithHyperParams net.FirstHiddenLayer bpSecHidLayer
 {
  net with
   OutputLayer = bpOutputLayer
   SecondHiddenLayer = bpSecHidLayer
   FirstHiddenLayer =  bpFirstHidLayer
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
  let training_rand_index = shuffle rand training_index
  let shuffled_training_s = dataAtIndex training_samples training_rand_index
  let shuffled_training_i = dataAtIndex teachingInputs training_rand_index

  let testing_index = [0..(testing_samples.Length - 1)]
  let testing_rand_index = shuffle rand testing_index
  let shuffled_testing_s = dataAtIndex testing_samples testing_rand_index
  let shuffled_testing_i = dataAtIndex testOutputs testing_rand_index

  let trained = List.fold2 (funcNet trainOnce) netAcc shuffled_training_s shuffled_training_i
  let rms_trained_err = networkDistance trained

  let validated = List.fold2 (funcNet feedForward) netAcc shuffled_testing_s shuffled_testing_i
  let rms_validated_err = networkDistance validated

  let log path data = File.AppendAllText(path, data)

  let logToDataFile filename =
   let fullfilepath = @"D:\Projects\AI\cardiotocography\script\"+filename
   log fullfilepath

//  (* write error *)
//  let errors trained_err validated_err = trained_err + "," + validated_err + "\n"
//  logToDataFile "errors.txt" <| errors (rms_trained_err.ToString()) (rms_validated_err.ToString())
//
//  (* write appropriate parameters.
//   -h1,h2,output weights and biases. *)
//  let logNetworkParameters =
//   (netAcc.FirstHiddenLayer.Weights |> matrixToString) + "," +
//   (netAcc.FirstHiddenLayer.Bias |> vectorToString) + "," +
//   (netAcc.SecondHiddenLayer.Weights |> matrixToString) + "," +
//   (netAcc.SecondHiddenLayer.Bias |> vectorToString) + "," +
//   (netAcc.OutputLayer.Weights |> matrixToString) + "," +
//   (netAcc.OutputLayer.Bias |> vectorToString) + "\n"
//  logToDataFile "weightsAndBiases.txt" <| logNetworkParameters

  if epoch % 100 = 0 then printfn "%f %f" rms_trained_err rms_validated_err
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
                     Gradients = List.empty
                     PrevDeltas = List.replicate hiddenSize <| List.replicate inputSize 0.0
                     BiasPrevDeltas = List.replicate hiddenSize 0.0
                     NetOutputs = List.empty
 }
 SecondHiddenLayer = {
                      Inputs = List.empty
                      Weights = ((*) hiddenSize hiddenSize) |> listRandElems |> List.chunkBySize hiddenSize
                      Bias = listRandElems hiddenSize
                      Gradients = List.empty
                      PrevDeltas = List.replicate hiddenSize <| List.replicate hiddenSize 0.0
                      BiasPrevDeltas = List.replicate hiddenSize 0.0
                      NetOutputs = List.empty
 }
 OutputLayer = {
                Inputs = List.empty
                Weights = ((*) hiddenSize outputSize) |> listRandElems |> List.chunkBySize hiddenSize
                Bias = listRandElems outputSize
                Gradients = List.empty
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

printfn "Training..."
let trained = train epoch network (training_samples, teaching_inputs) (testing_samples, test_outputs)