package crackTheCaptcha;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation; // defines different activation functions like RELU, SOFTMAX, etc.
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions; // mean squared error, multiclass cross entropy, etc.

public class TrainingTransfert {

	private static final long seed = 74456874;
	private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	private static final Random randNumGen = new Random(735122);

	
	/**
	 *  
 Accuracy:        0,9962
 Precision:       0,9962
 Recall:          0,9962
 F1 Score:        0,9961
	 */

	public static void main(String[] args) throws IOException {
		int height = 28;
		int width = 28;
		int channels = 1;
		int numEpochs = 10;

		File parentDir = new File("prep/train");
		String baseName = "model_transfert";
		MultiLayerNetwork networkOrigin = MultiLayerNetwork.load(new File("models/emnist2/bestModel.bin"), false);
		
		int classes = parentDir.list().length;
		
		if (networkOrigin != null) {
			System.out.println(networkOrigin.summary());
			System.out.println();
		}
		
		double inputRetainProbability = 0.99;
		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
		        
		        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		        .updater(new Nesterovs(0.0005, inputRetainProbability)) // learning rate, momentum
				.weightInit(WeightInit.XAVIER)
		        .seed(seed)
		        .build();
		
		
		MultiLayerNetwork network = new TransferLearning.Builder(networkOrigin)
        .fineTuneConfiguration(fineTuneConf)
        .setFeatureExtractor(networkOrigin.getLayer("dense1").getIndex())
        .removeOutputLayer()
        .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.name("new_output")
						.nIn(512)
						.nOut(classes)
						.activation(Activation.SOFTMAX)
						.build())
        .build();
		
		if (network != null) {
			System.out.println(network.summary());
			System.out.println();
		}
		
		
		
		
		FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

		InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
		InputSplit trainData = filesInDirSplit[0];
		InputSplit testData = filesInDirSplit[0];

		 ImageTransform transform = new MultiImageTransform(randNumGen,
		 new RotateImageTransform(10.f));

		//ImageTransform transform = new MultiImageTransform(randNumGen, new ShowImageTransform("Display - before "));

		ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler();

		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
		recordReader.initialize(trainData, transform);

		ImageRecordReader recordTestReader = new ImageRecordReader(height, width, channels, labelMaker);
		recordTestReader.initialize(testData);

		int outputNum = recordReader.numLabels();

		int batchSize = 64; // Minibatch size. Here: The number of images to fetch for each call to
								// dataIter.next().
		int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by
							// recordReader.next()

		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);
		dataIter.setPreProcessor(imagePreProcessingScaler);

		DataSetIterator dataTestIter = new RecordReaderDataSetIterator(recordTestReader, batchSize, labelIndex, outputNum);
		dataTestIter.setPreProcessor(imagePreProcessingScaler);

		/*
		 * while (dataIter.hasNext()) { var ds = dataIter.next();
		 * System.out.println(ds); try { Thread.sleep(3000); //1000 milliseconds is one
		 * second. } catch(InterruptedException ex) {
		 * Thread.currentThread().interrupt(); } }
		 */

		// pass a training listener that reports score every 10 iterations
		int listenerFrequency = 1;
		network.addListeners(new ScoreIterationListener(listenerFrequency));
		boolean reportScore = true;
		boolean reportGC = true;
		network.addListeners(new PerformanceListener(listenerFrequency, reportScore, reportGC));

		// StatsStorage statsStorage = new InMemoryStatsStorage();
		// uiServer.attach(statsStorage);
		// network.addListeners(new StatsListener(statsStorage, listenerFrequency));

		// var convolutionListener = new ConvolutionalIterationListener(eachIterations);
		// network.addListeners(convolutionListener);

		System.out.println("Training workspace config: " + network.getLayerWiseConfigurations().getTrainingWorkspaceMode());
		System.out.println("Inference workspace config: " + network.getLayerWiseConfigurations().getInferenceWorkspaceMode());

		// fit a dataset for a single epoch
		// network.fit(emnistTrain)

		// fit for multiple epochs
		// val numEpochs = 2
		// network.fit(emnistTrain, numEpochs)

		// or simply use for loop
		for (int i = 0; i < numEpochs; i++) {
			System.out.println("Epoch " + i + " / " + numEpochs);
			long start = System.currentTimeMillis();
			network.fit(dataIter);
			long end = System.currentTimeMillis();
			network.save(new File(baseName + ".r." + i + ".bin"));
			System.out.println("Epoch " + i + " / " + numEpochs + " -> " + ((end - start) / 1000) + "s");
		}
		/**/

		// network.save(new File("model_emnist_complete.bin"));

		// evaluate basic performance
		// var eval = network.evaluate[Evaluation](emnistTest);
		var eval = network.evaluate(dataTestIter);
		System.out.println(eval.accuracy());
		System.out.println(eval.precision());
		System.out.println(eval.recall());

		// evaluate ROC and calculate the Area Under Curve
		var roc = network.evaluateROCMultiClass(dataTestIter, 0);
		int classIndex = 1;
		roc.calculateAUC(classIndex);

		// optionally, you can print all stats from the evaluations
		System.out.print(eval.stats(false, true));
		System.out.print(roc.stats());

		System.out.println();
		
		
		System.out.println(" ------   end   ------");

		if (network != null) {
			System.out.println(network.summary());
			System.out.println();
		}
		
	}
	
}
