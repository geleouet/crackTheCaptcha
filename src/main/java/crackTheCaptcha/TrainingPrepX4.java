package crackTheCaptcha;

import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import com.twelvemonkeys.image.GrayFilter;

public class TrainingPrepX4 {

	
	public static void main(String[] args) throws IOException {
		
		var loadUpdater = false;
		System.out.println("Load");
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("trained_mnist_model.bin"), loadUpdater);
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modellarge.bin"), loadUpdater);
		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modelcomplete0.bin"), loadUpdater);
		
		
		ImageFilter filter = new GrayFilter();  

		BufferedImage read = ImageIO.read(new File("train/AMPB.png"));
		
		ImageProducer producer = new FilteredImageSource(read.getSource(), filter);  
		Image mage = Toolkit.getDefaultToolkit().createImage(producer);
		
		
		
		/*
		BufferedImageOp resampler = new ResampleOp(width, height, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
		BufferedImage output = resampler.filter(input, null);
		*/
		
		
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
