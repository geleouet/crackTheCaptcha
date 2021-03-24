package crackTheCaptcha;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.RotateImageTransform;

/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/



import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.Random;

/**
 * Description This is a demo that multi-digit number recognition. The maximum length is 6 digits.
 * If it is less than 6 digits, then zero is added to last
 * Training set: There were 14108 images, and they were used to train a model.
 * Testing set: in total 108 images,they copied from the training set,mainly to determine whether it's good that the model fited training data
 * Verification set: The total quantity of the images has 248 that's the unknown image,the main judgment that the model is good or bad
 * Other: Using the current architecture and hyperparameters, the accuracy of the best model prediction validation set is (215-227) / 248 with different epochs
 * of course, if you're interested, you can continue to optimize.
 * @author WangFeng
 */
public class MultiPass {

    private static final Logger log = LoggerFactory.getLogger(MultiPass.class);

    private static int batchSize = 15;
    private static String rootPath = "multi";

    private static String modelDirPath = rootPath + File.separatorChar + "out" + File.separatorChar + "models";
    private static String modelPath = modelDirPath + File.separatorChar + "validateCodeCheckModel.json";

    
    
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	private static final Random randNumGen = new Random(735122);
	public static class FileLabelGenerator implements PathLabelGenerator {


	    @Override
	    public Writable getLabelForPath(String path) {
	        // Label is in the directory
	        String dirName = new File(path).getName().split("\\.")[0];
	        return new Text(dirName);
	    }

	    @Override
	    public Writable getLabelForPath(URI uri) {
	        return getLabelForPath(new File(uri).toString());
	    }

	    @Override
	    public boolean inferLabelClasses() {
	        return true;
	    }
	}
    
    public static void mains(String[] args) throws IOException {
		int height = 28;
		int width = 28;
		int channels = 1;
		int numEpochs = 10;

		File parentDir = new File("prep/train");
		int classes = parentDir.list().length;
		var network = createModel(numEpochs);
		//var network = MultiLayerNetwork.load(new File("bestModel.bin"), false);
		network.init();

		FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
		FileLabelGenerator labelMaker = new FileLabelGenerator();
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
    
    
    
    public static void main(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();
        System.out.println(startTime);

        File modelDir = new File(modelDirPath);

        // create directory
        if (!modelDir.exists()) { //noinspection ResultOfMethodCallIgnored
            modelDir.mkdirs(); }
        log.info( modelPath );
        //create model
        ComputationGraph model =  createModel(62);
        //monitor the model score
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File("multi", "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        //construct the iterator
        MultiDataSetIterator trainMulIterator = new MultiRecordDataSetIterator(batchSize, "train");
        MultiDataSetIterator testMulIterator = new MultiRecordDataSetIterator(batchSize,"test");
        MultiDataSetIterator validateMulIterator = new MultiRecordDataSetIterator(batchSize,"validate");

        //fit
        model.setListeners(new ScoreIterationListener(10), new StatsListener( statsStorage), new EvaluativeListener(testMulIterator, 1, InvocationType.EPOCH_END));
        int epochs = 4;
        model.fit(trainMulIterator, epochs);

        //save
        model.save(new File(modelPath), true);
        long endTime = System.currentTimeMillis();

        System.out.println("=============run time=====================" + (endTime - startTime));

        System.out.println("=====eval model=====test==================");
        modelPredict(model, testMulIterator);

        System.out.println("=====eval model=====validate==================");
        modelPredict(model, validateMulIterator);

    }

    private static ComputationGraph createModel(int nClasses) {
    	double inputRetainProbability = 0.8;
		var config = new NeuralNetConfiguration.Builder()
				.seed(146534313)
				//.l2(0.0005) // ridge regression value
				.updater(new Nesterovs(0.005, inputRetainProbability)) // learning rate, momentum
				.weightInit(WeightInit.XAVIER)

				.cacheMode(CacheMode.HOST)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.graphBuilder()
				.setOutputs("output1", "output2", "output3", "output4")
				.addInputs("input")
				.layer("conv1", new ConvolutionLayer.Builder(5, 5)//5, 5
						.nIn(1)
						.stride(1, 1)
						.nOut(48)
						.activation(Activation.RELU)
						.convolutionMode(ConvolutionMode.Same)
						.build(), "input")
				.layer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.dropOut(inputRetainProbability)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build(), "conv1")
				
				.layer("conv2", new ConvolutionLayer.Builder(5, 5)//5, 5
						.stride(1, 1)
						.nOut(64)
						.activation(Activation.RELU)
						.convolutionMode(ConvolutionMode.Same)
						.build(), "pool1")
				.layer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.name("pool2")
						.kernelSize(2, 1)
						.stride(2, 1)
						.build(), "conv2")
				
				.layer("conv3", new ConvolutionLayer.Builder(3, 3)
						.stride(1, 1) // nIn need not specified in later layers
						.nOut(128)
						.activation(Activation.RELU)
						.build(), "pool2")
				.layer("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build(), "conv3")

				.layer("conv4", new ConvolutionLayer.Builder(3, 3)
						.stride(1, 1) // nIn need not specified in later layers
						.nOut(256)
						.activation(Activation.RELU)
						.build(), "pool3")
				.layer("pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build(), "conv4")
				
				.layer("dense1", new DenseLayer.Builder().activation(Activation.RELU)
						.dropOut(inputRetainProbability)
						.nOut(3072)
						.build(), "pool4")
				.layer("dense2", new DenseLayer.Builder().activation(Activation.RELU)
						.dropOut(inputRetainProbability)
						.nOut(3072)
						.build(), "dense1")
				.layer("output1", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(nClasses)
						.activation(Activation.SOFTMAX)
						.build(), "dense2")
				.layer("output2", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(nClasses)
						.activation(Activation.SOFTMAX)
						.build(), "dense2")
				.layer("output3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(nClasses)
						.activation(Activation.SOFTMAX)
						.build(), "dense2")
				.layer("output4", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(nClasses)
						.activation(Activation.SOFTMAX)
						.build(), "dense2")
				.setInputTypes(InputType.convolutionalFlat(50, 150, 1)) // InputType.convolutional for normal image
				.build();
    	
    	/*
    	
        long seed = 123;
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(seed)
            //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            //.l2(1e-3)
            //.updater(new Adam(1e-3))
			.cacheMode(CacheMode.HOST)
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			
            .weightInit( WeightInit.XAVIER)
            .graphBuilder()
            .addInputs("trainFeatures")
            .setInputTypes(InputType.convolutional(50, 150, 1))
            .setOutputs("out1", "out2", "out3", "out4")
            .addLayer("cnn1",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nIn(1).nOut(48).activation( Activation.RELU).convolutionMode(ConvolutionMode.Same).build(), "trainFeatures")
            .addLayer("maxpool1",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn1")
            .addLayer("cnn2",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(64).activation( Activation.RELU).convolutionMode(ConvolutionMode.Same).build(), "maxpool1")
            .addLayer("maxpool2",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,1}, new int[]{2, 1}, new int[]{0, 0})
                .build(), "cnn2")
            .addLayer("cnn3",  new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(128).activation( Activation.RELU).build(), "maxpool2")
            .addLayer("maxpool3",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn3")
            .addLayer("cnn4",  new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(256).activation( Activation.RELU).build(), "maxpool3")
            .addLayer("maxpool4",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn4")
            .addLayer("ffn0",  new DenseLayer.Builder().nOut(3072)
                .build(), "maxpool4")
            .addLayer("ffn1",  new DenseLayer.Builder().nOut(3072)
                .build(), "ffn0")
            .addLayer("out1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nClasses).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out2", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nClasses).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out3", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nClasses).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out4", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nClasses).activation(Activation.SOFTMAX).build(), "ffn1")
            //.addLayer("out5", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            //    .nOut(nClasses).activation(Activation.SOFTMAX).build(), "ffn1")
            //.addLayer("out6", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            //    .nOut(nClasses).activation(Activation.SOFTMAX).build(), "ffn1")
            .build();
    	 */
        // Construct and initialize model
        ComputationGraph model = new ComputationGraph(config);
        model.init();
        System.out.println(model.summary());

        return model;
    }

    private static void modelPredict(ComputationGraph model, MultiDataSetIterator iterator) {
        int sumCount = 0;
        int correctCount = 0;

        while (iterator.hasNext()) {
            MultiDataSet mds = iterator.next();
            INDArray[]  output = model.output(mds.getFeatures());
            INDArray[] labels = mds.getLabels();
            int dataNum = Math.min(batchSize, output[0].rows());
            for (int dataIndex = 0;  dataIndex < dataNum; dataIndex ++) {
                StringBuilder reLabel = new StringBuilder();
                StringBuilder peLabel = new StringBuilder();
                INDArray preOutput;
                INDArray realLabel;
                for (int digit = 0; digit < 6; digit ++) {
                    preOutput = output[digit].getRow(dataIndex);
                    peLabel.append(Nd4j.argMax(preOutput).getInt(0));
                    realLabel = labels[digit].getRow(dataIndex);
                    reLabel.append(Nd4j.argMax(realLabel).getInt(0));
                }
                boolean equals = peLabel.toString().equals(reLabel.toString());
                if (equals) {
                    correctCount ++;
                }
                sumCount ++;
                log.info("real image {}  prediction {} status {}", reLabel.toString(), peLabel.toString(), equals);
            }
        }
        iterator.reset();
        System.out.println("validate result : sum count =" + sumCount + " correct count=" + correctCount );
    }
}