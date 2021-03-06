package crackTheCaptcha;



import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation; // defines different activation functions like RELU, SOFTMAX, etc.
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions; // mean squared error, multiclass cross entropy, etc.

/**
 * 
 * https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
 * https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
 * 
 * https://deeplearning4j.konduit.ai/getting-started/beginners
 * https://medium.com/datactw/deep-learning-for-java-dl4j-getting-started-tutorial-2259c76c0a7c
 * https://www.rcp-vision.com/convolutional-neural-networks-with-eclipse-deeplearning4j/
 * 
 * https://www.dubs.tech/guides/quickstart-with-dl4j/
 * 
 * @author gaeta
 *
 */
public class CrackTheCaptcha {

	public static void main(String[] args) throws IOException {
		new CrackTheCaptcha().start();
	}

	private void start() throws IOException {
		//UIServer uiServer = UIServer.getInstance();

		int batchSize = 512; // how many examples to simultaneously train in the network /128 en local
		var emnistSet = EmnistDataSetIterator.Set.COMPLETE;
		EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
		EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);

		/*var scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(emnistTrain);
		emnistTrain.setPreProcessor(scaler);
		emnistTest.setPreProcessor(scaler);*/
		
		var numRows = 28; // number of "pixel rows" in an mnist digit
		var numColumns = 28;
		var channels = 1;
		
		//emnistTrain.setPreProcessor(new CropAndResizeDataSetPreProcessor(28, 28, 0, 0, numRows, numColumns, channels, ResizeMethod.Bilinear));
		
		int numEpochs = 1000;
		
		List<String> labels = emnistTrain.getLabels();
		for (int i = 0; i < labels.size(); i++) {
			System.out.println(i + " " + labels.get(i));
		}
		
		var outputNum = EmnistDataSetIterator.numLabels(emnistSet); // total output classes
		var rngSeed = 123; // integer for reproducability of a random number generator
		
		MultiLayerConfiguration conf = networkConfiguration(numRows, numColumns, channels, outputNum, rngSeed);
		
		
		/**/
		EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
		        .epochTerminationConditions(new MaxEpochsTerminationCondition(numEpochs), new ScoreImprovementEpochTerminationCondition(5))
		        .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(600, TimeUnit.MINUTES))
		        .scoreCalculator(new DataSetLossCalculator(emnistTest, true))
		        .evaluateEveryNEpochs(1)
		        .modelSaver(new LocalFileModelSaver("."))
		        .build();

		var network = new MultiLayerNetwork(conf);
		network.init();

		// pass a training listener that reports score every 10 iterations
		int listenerFrequency = 100;
		network.addListeners(new ScoreIterationListener(listenerFrequency));
		boolean reportScore = true;
		boolean reportGC = true;
		network.addListeners(new PerformanceListener(listenerFrequency, reportScore, reportGC));

		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, network, emnistTrain);

		//Conduct early stopping training:
		EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

		//Print out the results:
		System.out.println("Termination reason: " + result.getTerminationReason());
		System.out.println("Termination details: " + result.getTerminationDetails());
		System.out.println("Total epochs: " + result.getTotalEpochs());
		System.out.println("Best epoch number: " + result.getBestModelEpoch());
		System.out.println("Score at best epoch: " + result.getBestModelScore());

		//Get the best model:
		MultiLayerNetwork bestModel = result.getBestModel();
		
		/*
		// create the MLN
		var network = new MultiLayerNetwork(conf);
		network.init();

		// pass a training listener that reports score every 10 iterations
		int listenerFrequency = 100;
		network.addListeners(new ScoreIterationListener(listenerFrequency));
		boolean reportScore = true;
		boolean reportGC = true;
		network.addListeners(new PerformanceListener(listenerFrequency, reportScore, reportGC));
		
		//StatsStorage statsStorage = new InMemoryStatsStorage();
		//uiServer.attach(statsStorage);
		//network.addListeners(new StatsListener(statsStorage, listenerFrequency));
		
		//var convolutionListener = new ConvolutionalIterationListener(eachIterations);
		//network.addListeners(convolutionListener);

		System.out.println("Training workspace config: " + network.getLayerWiseConfigurations().getTrainingWorkspaceMode());
		System.out.println("Inference workspace config: " + network.getLayerWiseConfigurations().getInferenceWorkspaceMode());


		// fit a dataset for a single epoch
		// network.fit(emnistTrain)

		// fit for multiple epochs
		// val numEpochs = 2
		// network.fit(emnistTrain, numEpochs)

		// or simply use for loop
		for(int i = 0; i < numEpochs; i++) {
		   System.out.println("Epoch " + i + " / " + numEpochs);
		   long start = System.currentTimeMillis();
		   network.fit(emnistTrain);
		   long end = System.currentTimeMillis();
		   network.save(new File("model_emnist_complete."+i+".bin"));
		   System.out.println("Epoch " + i + " / " + numEpochs + " -> " + ((end - start) / 1000) +"s");
		}
		/**/
		
		//network.save(new File("model_emnist_complete.bin"));
		
		// evaluate basic performance
		//var eval = network.evaluate[Evaluation](emnistTest);
		var eval = bestModel.evaluate(emnistTest);
		System.out.println(eval.accuracy());
		System.out.println(eval.precision());
		System.out.println(eval.recall());

		// evaluate ROC and calculate the Area Under Curve
		var roc = bestModel.evaluateROCMultiClass(emnistTest, 0);
		int classIndex = 1;
		roc.calculateAUC(classIndex);

		// optionally, you can print all stats from the evaluations
		System.out.print(eval.stats(false, true));
		System.out.print(roc.stats());
		
		System.out.println(" ------   end   ------");
	}

	private MultiLayerConfiguration networkConfiguration(int numRows, int numColumns, int channels, int outputNum, int rngSeed) {
		double inputRetainProbability = 0.8;
		return new NeuralNetConfiguration.Builder()
				.seed(rngSeed)
				.l2(0.0005) // ridge regression value
				.updater(new Nesterovs(0.0005, inputRetainProbability)) // learning rate, momentum
				.weightInit(WeightInit.XAVIER)

				.cacheMode(CacheMode.HOST)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5)//5, 5
						.name("conv1")
						.dropOut(inputRetainProbability)
						.nIn(channels)
						.stride(1, 1)
						.nOut(64)  //20
						.activation(Activation.RELU)
						.convolutionMode(ConvolutionMode.Same)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.name("pool1")
						.dropOut(inputRetainProbability)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)
						.name("conv2")
						.dropOut(inputRetainProbability)
						.stride(1, 1) // nIn need not specified in later layers
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.name("pool2")
						.dropOut(inputRetainProbability)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)
						.name("conv3")
						.dropOut(inputRetainProbability)
						.stride(1, 1) // nIn need not specified in later layers
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.name("pool3")
						.dropOut(inputRetainProbability)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder().activation(Activation.RELU)
						.name("dense1")
						.dropOut(inputRetainProbability)
						.nOut(512)
						.build())
				.layer(new DenseLayer.Builder().activation(Activation.RELU)
						.name("dense2")
						.dropOut(inputRetainProbability)
						.nOut(512)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.name("output")
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutionalFlat(numRows, numColumns, channels)) // InputType.convolutional for normal image
				.build();
	}
	
}
