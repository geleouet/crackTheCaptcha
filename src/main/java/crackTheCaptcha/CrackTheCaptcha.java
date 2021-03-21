package crackTheCaptcha;



import java.io.File;
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.weights.ConvolutionalIterationListener;
import org.nd4j.linalg.activations.Activation; // defines different activation functions like RELU, SOFTMAX, etc.
import org.nd4j.linalg.dataset.api.preprocessor.CropAndResizeDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.CropAndResizeDataSetPreProcessor.ResizeMethod;
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
		UIServer uiServer = UIServer.getInstance();

		int batchSize = 256; // how many examples to simultaneously train in the network
		var emnistSet = EmnistDataSetIterator.Set.COMPLETE;
		EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
		EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);

		var numRows = 28; // number of "pixel rows" in an mnist digit
		var numColumns = 28;
		var channels = 1;
		
		//emnistTrain.setPreProcessor(new CropAndResizeDataSetPreProcessor(28, 28, 0, 0, numRows, numColumns, channels, ResizeMethod.Bilinear));
		
		int numEpochs = 10;
		
		List<String> labels = emnistTrain.getLabels();
		for (int i = 0; i < labels.size(); i++) {
			System.out.println(i + " " + labels.get(i));
		}
		
		var outputNum = EmnistDataSetIterator.numLabels(emnistSet); // total output classes
		var rngSeed = 123; // integer for reproducability of a random number generator
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs(0.005, 0.9)) // learning rate, momentum
                .weightInit(WeightInit.XAVIER)
                
                .cacheMode(CacheMode.HOST)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)//5, 5
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)  //20
                    .activation(Activation.RELU)
                    .convolutionMode(ConvolutionMode.Same)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.RELU)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build())
                .setInputType(InputType.convolutionalFlat(numRows, numColumns, channels)) // InputType.convolutional for normal image
                .build();
		
		// create the MLN
		var network = new MultiLayerNetwork(conf);
		network.init();

		// pass a training listener that reports score every 10 iterations
		int listenerFrequency = 100;
		network.addListeners(new ScoreIterationListener(listenerFrequency));
		boolean reportScore = true;
		boolean reportGC = true;
		network.addListeners(new PerformanceListener(listenerFrequency, reportScore, reportGC));
		
		
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		network.addListeners(new StatsListener(statsStorage, listenerFrequency));
		
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
		
		
		network.save(new File("model_emnist_complete.bin"));
		
		// evaluate basic performance
		//var eval = network.evaluate[Evaluation](emnistTest);
		var eval = network.evaluate(emnistTest);
		System.out.println(eval.accuracy());
		System.out.println(eval.precision());
		System.out.println(eval.recall());

		// evaluate ROC and calculate the Area Under Curve
		var roc = network.evaluateROCMultiClass(emnistTest, 0);
		int classIndex = 1;
		roc.calculateAUC(classIndex);

		// optionally, you can print all stats from the evaluations
		System.out.print(eval.stats(false, true));
		System.out.print(roc.stats());
		
		System.out.println(" ------   end   ------");
	}
	
}
