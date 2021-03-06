package crackTheCaptcha;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.stream.Collectors;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class TrainingPrep {

	
	public static void main(String[] args) throws IOException {
		// Create Captcha container
		OxCaptcha c = new OxCaptcha(150, 50);

		// Create background
		c.background();
		//c.backgroundFlat();
		//c.backgroundGradient();
		//c.backgroundSquiggles();

		// Add text
		c.text("0");
		//c.text("2a2ba");
		//c.text(new char[] {'a', 'b', 'c'}, new int[] {1, 2, 7}, new int[] {30, 5, -10});


		// Add noise
		//c.noise();
		//c.noiseStraightLine();
		//c.noiseCurvedLine();

		// Apply transformation
		//c.transformFishEye();
		//c.transformStretch();
		//c.transformShear();
		c.distortionElastic();

		// Get rendered image
		//BufferedImage img = c.getImage();

		// Get rendered image as a 2D array
		//int[][] imgArray = c.getImageArray2D();

		// Write image to a png file
		//c.save("training/0u.png");
		
		var loadUpdater = false;
		System.out.println("Load");
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("trained_mnist_model.bin"), loadUpdater);
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modellarge.bin"), loadUpdater);
		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modelcomplete0.bin"), loadUpdater);
		
		
		int height = 28;
		int width = 28;
		int channels = 1;
		NativeImageLoader loader = new NativeImageLoader(height, width, channels);
		//File f=new File("mnist_png/training/5/img_389.jpg");
		//File f=new File("mnist_png/training/4/img_329.jpg");
		File f=new File("training/A.png");
		//put image into INDArray
		INDArray image = loader.asMatrix(f);

		//values need to be scaled
		DataNormalization scalar = new ImagePreProcessingScaler(0, 1);

		// then call that scalar on the image dataset
		scalar.transform(image);

		System.out.println("Start");
		
		INDArray result = net2.output(image);
		

		//used to control the number of decimals places for the output probability
        DecimalFormat df2 = new DecimalFormat(".##");

        //transfer the neural network output to an array
        double[] results = {result.getDouble(0,0),result.getDouble(0,1),result.getDouble(0,2),
                result.getDouble(0,3),result.getDouble(0,4),result.getDouble(0,5),result.getDouble(0,6),
                result.getDouble(0,7),result.getDouble(0,8),result.getDouble(0,9),result.getDouble(0,10),result.getDouble(0,11),};

        //find the UI tvs to display the prediction and confidence values

        //display the values using helper functions defined below
        System.out.println(String.valueOf(df2.format(arrayMaximum(results))));
        System.out.println(String.valueOf(getIndexOfLargestValue(results)));

		
		int[] predict = net2.predict(image);
		
		System.out.println(Arrays.stream(predict).mapToObj(i -> EmnistDataSetIterator.getLabels(EmnistDataSetIterator.Set.BALANCED).get(i)).collect(Collectors.joining(", ")));
	}
	
	//helper class to return the largest value in the output array
    public static double arrayMaximum(double[] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for(double cur: arr)
            max = Math.max(max, cur);
        return max;
    }

    // helper class to find the index (and therefore numerical value) of the largest confidence score
    public static int getIndexOfLargestValue( double[] array )
    {
        if ( array == null || array.length == 0 ) return -1;
        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {if ( array[i] > array[largest] ) largest = i;            }
        return largest;
    }
//	protected INDArray load(String... params) {
//        // Main background thread, this will load the model and test the input image
//    // The dimensions of the images are set here
//        int height = 28;
//        int width = 28;
//        int channels = 1;
//
//        //Now we load the model from the raw folder with a try / catch block
//        try {
//            // Load the pretrained network.
//            InputStream inputStream = getResources().openRawResource(R.raw.trained_mnist_model);
//            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(inputStream);
//
//            //load the image file to test
//            File f=new File("training/A.png");
//
//            //Use the nativeImageLoader to convert to numerical matrix
//            NativeImageLoader loader = new NativeImageLoader(height, width, channels);
//
//            //put image into INDArray
//            INDArray image = loader.asMatrix(f);
//
//            //values need to be scaled
//            DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
//
//            //then call that scalar on the image dataset
//            scalar.transform(image);
//
//            //pass through neural net and store it in output array
//            output = model.output(image);
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        return output;
//    }
	
}
