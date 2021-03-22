package crackTheCaptcha;

import static org.bytedeco.opencv.global.opencv_core.BORDER_REPLICATE;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_SHAPE_ELLIPSE;
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_EXTERNAL;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_OTSU;
import static org.bytedeco.opencv.global.opencv_imgproc.boundingRect;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.findContours;
import static org.bytedeco.opencv.global.opencv_imgproc.getStructuringElement;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.DataBufferByte;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import com.twelvemonkeys.image.BufferedImageFactory;
import com.twelvemonkeys.image.GrayFilter;
import com.twelvemonkeys.image.ResampleOp;

/**
 * https://stackoverflow.com/questions/48845162/how-can-i-use-a-custom-data-model-with-deeplearning4j
 * 
 * @author gaeta
 *
 */
public class TrainingPrepX4 {

	static boolean debugNet = false;
	static boolean debugPrepare = false;
	
	public static void main(String[] args) throws IOException {
		int width = 28;
		int height = 28;
		
		debugPrepare = true;
		debugNet = true;
		
		ImageFilter filter = new GrayFilter();  
		BufferedImageOp resampler = new ResampleOp(width, height, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
		
		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("bestModel.bin"), false);
		GuessResult analyse = analyse(width, height, net2, filter, resampler, "uACyk.jpeg");
		System.out.println(analyse.guess + ";"+analyse.confident);
	}
	public static void mainAll(String[] args) throws IOException {
		
		var loadUpdater = false;
		int width = 28;
		int height = 28;
		
		
		System.out.println("Load");
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("trained_mnist_model.bin"), loadUpdater);
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modellarge.bin"), loadUpdater);
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modelcomplete0.bin"), loadUpdater);

		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("emnist_model.bin"), loadUpdater);
		ImageFilter filter = new GrayFilter();  
		BufferedImageOp resampler = new ResampleOp(width, height, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
		
		
		File evalDirectory = new File("train");

		int correct = 0;
		int all = 0;
		for (File f : evalDirectory.listFiles()) {
			String reference = f.getName().substring(0, 4);
			
			GuessResult analyse = analyse(width, height, net2, filter, resampler, f.getAbsolutePath());
			if (reference.equals(analyse.guess)) {
				correct++;
			}
			all++;
			
			System.out.println(analyse.guess + ";" + reference+";"+ analyse.confident);
		}
		
		System.out.println("Captcha " + correct  + "/" + all);
		
	}

	private static GuessResult analyse(int width, int height, MultiLayerNetwork net2, ImageFilter filter, BufferedImageOp resampler, String fileName) throws IOException {
		BufferedImage read = ImageIO.read(new File(fileName));
		int original_width = read.getWidth();
		int original_height = read.getHeight();
		
		ImageProducer producer = new FilteredImageSource(read.getSource(), filter);  
		Image mage = Toolkit.getDefaultToolkit().createImage(producer);
		
		Mat origin  = imread(fileName);
		Mat grayScale = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat grayScaleHist = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat grayScaleWthBorder = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat thresholded = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat eroded = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat dilated = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat hierarchy = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		
		cvtColor(origin, grayScale, COLOR_BGR2GRAY);
		if (debugPrepare) imwrite("grayscale.png", grayScale);
		
		threshold(grayScale, grayScaleHist, 128, 255, THRESH_OTSU);
		//equalizeHist(grayScale, grayScaleHist);
		if (debugPrepare) imwrite("grayscalehist.png", grayScaleHist);
		
		
		copyMakeBorder(grayScaleHist, grayScaleWthBorder, 8, 8, 8, 8, BORDER_REPLICATE);
		if (debugPrepare) imwrite("grayScaleWthBorder.png", grayScaleWthBorder);
		
		threshold(grayScaleWthBorder, thresholded, 128, 255, THRESH_BINARY_INV | THRESH_OTSU);
		if (debugPrepare) imwrite("thresholded.png", thresholded);
		
		int kernelSize = 2;
		var elementType = CV_SHAPE_ELLIPSE;
        Mat element = getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));
        erode(thresholded, eroded, element);
        if (debugPrepare) imwrite("eroded.png", eroded);
		
        dilate(eroded, dilated, element);
        if (debugPrepare) imwrite("dilated.png", dilated);
        
		MatVector contours = new MatVector();
		findContours(dilated, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		
		
		BufferedImage boxedDbg = debugPrepare ? ImageIO.read(new File(fileName)) : null;
		
		
		List<Rect> boxes = new ArrayList<>();
		for (int idx = 0; idx < contours.size(); idx ++) {
			Mat p = contours.get(idx);
			Rect rect = boundingRect(p);
			if (rect.area() < 64) continue;
			
			if ((0.001 + rect.width()) / rect.height() > 1.25) {
				Rect a = new Rect(rect.x(), rect.y(), rect.width()/2+1, rect.height());
				Rect b = new Rect(rect.x()+rect.width()/2, rect.y(), rect.width() - rect.width()/2+1, rect.height());
				boxes.add(a);
				boxes.add(b);
			}
			else {
				boxes.add(rect);
			}
		}
		Map<Integer, GuessResult> guessed = new HashMap<>();
		for (int idx = 0; idx < boxes.size(); idx ++) {
			Rect rect = boxes.get(idx);
			//System.out.println(rect.x() + " " + rect.y()+ " " + rect.width() + " " + rect.height());
			//BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(Math.max(rect.x()-8, 0), Math.max(0, rect.y()-8), Math.min(original_width - Math.max(rect.x()-8, 0), rect.width()),
			//		Math.min(original_height - Math.max(rect.y()-8, 0), rect.height()));
			BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(
					Math.max(rect.x()-8 -1 -kernelSize, 0), 
					Math.max(0, rect.y()-8 -1 -kernelSize), 
					Math.min(original_width - Math.max(rect.x()-8 -1-kernelSize, 0), rect.width()+2+2*kernelSize),
					Math.min(original_height - Math.max(rect.y()-8-1-kernelSize, 0), rect.height()+2+2*kernelSize));
			if (debugPrepare) ImageIO.write(cropped, "PNG", new File("cropped_"+idx+".png"));
			
			BufferedImage output = resampler.filter(cropped, null);
			if (debugPrepare) ImageIO.write(output, "PNG", new File("output_"+idx+".png"));
			
			if (debugPrepare) {
				Graphics graphics = boxedDbg.getGraphics();
				graphics.setColor(Color.red);
				System.out.println(Math.max(rect.x()-8-kernelSize, 0) +", "+ Math.max(0, rect.y()-8-kernelSize)+", "+ Math.min(original_width - Math.max(rect.x()-8-kernelSize, 0), rect.width()+2*kernelSize)+", "+
						Math.min(original_height - Math.max(rect.y()-8-kernelSize, 0), rect.height()+2*kernelSize));	
				graphics.drawRect(Math.max(rect.x()-8-kernelSize, 0), Math.max(0, rect.y()-8-kernelSize), Math.min(original_width - Math.max(rect.x()-8-kernelSize, 0), rect.width()+2*kernelSize),
						Math.min(original_height - Math.max(rect.y()-8-kernelSize, 0), rect.height()+2*kernelSize));	
			}
			
			GuessResult g = net2 !=  null ? guess(width, height, net2, invert(output)) : new GuessResult("", 0.);
			if (g.confident > 0.2) guessed.put(rect.x(), g);
			
			if (debugNet) {
				System.out.println(idx + ":" + g.guess + " (" + g.confident + ")");
			}
			
		}
		
		if (debugPrepare) {
			ImageIO.write(boxedDbg, "PNG", new File("output_boxed.png"));
			
		}
		String r = guessed.entrySet().stream().sorted(Comparator.comparing(e -> e.getKey())).map(e -> e.getValue().guess).collect(Collectors.joining());
		double c = guessed.entrySet().stream().sorted(Comparator.comparing(e -> e.getKey())).mapToDouble(e -> e.getValue().confident).reduce(1, (a,b) -> a*b);
		return new GuessResult(r, c);
	}

	private static BufferedImage invert(BufferedImage output) {
		BufferedImage image = new BufferedImage(output.getWidth(), output.getHeight(),  
				BufferedImage.TYPE_BYTE_GRAY);  
		Graphics g = image.getGraphics();  
		g.drawImage(output, 0, 0, null);  
		g.dispose(); 

		invertRaster(image.getRaster());
		return image;
	}

	static class GuessResult {
		String guess;
		double confident;
		
		public GuessResult(String guess, double confident) {
			super();
			this.guess = guess;
			this.confident = confident;
		}
		
	}
	
	private static GuessResult guess(int width, int height, MultiLayerNetwork net2, BufferedImage output) throws IOException {
		int channels = 1;
		var loader = new Java2DNativeImageLoader(height, width, channels);
		//NativeImageLoader loader = new NativeImageLoader(height, width, channels);
		//File f=new File("mnist_png/training/5/img_389.jpg");
		//File f=new File("mnist_png/training/4/img_329.jpg");
		//File f=new File("training/A.png");
		//put image into INDArray
		//INDArray image = loader.asMatrix(f);
		
		INDArray image = loader.asMatrix(output);

		//values need to be scaled
		DataNormalization scalar = new ImagePreProcessingScaler(0, 1);

		// then call that scalar on the image dataset
		scalar.transform(image);

		INDArray result = net2.output(image);
		


        int nClasses = (int) result.data().length();
        List<String> labels = EmnistDataSetIterator.getLabels(EmnistDataSetIterator.Set.COMPLETE);
        double[] results = new double[nClasses];
        //transfer the neural network output to an array
        for (int i = 0; i < nClasses; i++) {results[i] = result.getDouble(0, i);}
        
        //used to control the number of decimals places for the output probability
//        DecimalFormat df2 = new DecimalFormat("0.##");
//        for (int i = 0; i < nClasses; i++) {
//        	if (results[i]>0.01) System.out.println(df2.format(results[i]) + ":" + labels.get(i));
//		}

        //display the values using helper functions defined below
        double maximum = arrayMaximum(results);
        String guess = labels.get(getIndexOfLargestValue(results));
        
//		System.out.println("--------" + guess + " @ "+ String.valueOf(df2.format(maximum)));
//		System.out.println();
        
        return new GuessResult(guess, maximum);
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
    
    static void invertRaster(final Raster raster) {
	    byte[] data = ((DataBufferByte) raster.getDataBuffer()).getData();

	    for (int i = 0, dataLength = data.length; i < dataLength; i++) {
	        data[i] = (byte) (255 - data[i] & 0xff);
	    }
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
