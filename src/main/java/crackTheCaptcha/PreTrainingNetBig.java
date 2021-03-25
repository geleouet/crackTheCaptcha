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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class PreTrainingNetBig {

	private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	private static final Random randNumGen = new Random(735122);

	public static void main(String[] args) throws IOException {
		int height = 28;
		int width = 28;
		int channels = 1;
		int numEpochs = 10;

		File parentDir = new File("ox/train");
		int classes = parentDir.list().length;
		var conf = networkConfiguration(width, height, 1, classes, 3289322);
		var network = new MultiLayerNetwork(conf);
		//var network = MultiLayerNetwork.load(new File("bestModel.bin"), false);
		network.init();

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
			network.save(new File("model_custom.r." + i + ".bin"));
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
		//var roc = network.evaluateROCMultiClass(dataTestIter, 0);
		//int classIndex = 1;
		//roc.calculateAUC(classIndex);

		// optionally, you can print all stats from the evaluations
		//System.out.print(eval.stats(false, true));
		//System.out.print(roc.stats());

		System.out.println(" ------   end   ------");
	}

	private static MultiLayerConfiguration networkConfiguration(int numRows, int numColumns, int channels, int outputNum, int rngSeed) {
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
