package crackTheCaptcha;

import static org.bytedeco.opencv.global.opencv_core.BORDER_REPLICATE;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;
import static org.bytedeco.opencv.global.opencv_core.inRange;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_SHAPE_ELLIPSE;
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_EXTERNAL;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_OTSU;
import static org.bytedeco.opencv.global.opencv_imgproc.boundingRect;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.erode;
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
import java.awt.image.RescaleOp;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
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
	
	//static List<String> labels = EmnistDataSetIterator.getLabels(EmnistDataSetIterator.Set.COMPLETE);
	static List<String> labels = Arrays.asList("A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(","));
	
	public static void maindbg(String[] args) throws IOException {
		int width = 28;
		int height = 28;
		
		debugPrepare = true;
		debugNet = true;
		
		ImageFilter filter = new GrayFilter();  
		BufferedImageOp resampler = new ResampleOp(width, height, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
		
		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("model_transfert.r.9.bin"), false);
//		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("bestModel.bin"), false);
		//GuessResult analyse = analyse(width, height, net2, filter, resampler, "test/YRBO.png");
		GuessResult analyse = analyse(width, height, net2, filter, resampler, "ZMU6.png");
		//GuessResult analyse = analyse(width, height, net2, filter, resampler, "Q3OT.png");
		
//		GuessResult analyse = analyse(width, height, net2, filter, resampler, "uACyk.jpeg");
		System.out.println(analyse.guess + ";"+analyse.confident);
	}
	public static void main(String[] args) throws IOException {
		
		var loadUpdater = false;
		int width = 28;
		int height = 28;
		
		
		System.out.println("Load");
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("trained_mnist_model.bin"), loadUpdater);
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modellarge.bin"), loadUpdater);
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("modelcomplete0.bin"), loadUpdater);

		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("model_custom.r.9.bin"), loadUpdater);
		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("model_transfert.r.9.bin"), loadUpdater);
		ImageFilter filter = new GrayFilter();  
		BufferedImageOp resampler = new ResampleOp(width, height, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
		
		
		
		File evalDirectory = new File("datas/train");
		File errorDirectory = new File("error");
		if (!errorDirectory.exists()) errorDirectory.mkdirs();
		for (File f :errorDirectory.listFiles()) f.delete();

		if (!new File("dbg").exists()) new File("dbg").mkdirs(); 
		for (File f : new File("dbg").listFiles()) f.delete();

		int correct = 0;
		int all = 0;
		for (File f : evalDirectory.listFiles()) {
			String reference = f.getName().split("")[0];
			
			GuessResult analyse = analyse(width, height, net2, filter, resampler, f.getAbsolutePath());
			
			
			if (reference.equals(analyse.guess)) {
				correct++;
			}
			else {
				System.out.println(analyse.guess + ";" + reference+";"+ analyse.confident);
				var split = f.getName().split("\\.");
				Files.copy(f.toPath(), Path.of(errorDirectory.getAbsolutePath(), split[0] + "_" + analyse.guess+"."+split[1]));
			}
			all++;
			
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
		Mat inRange = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat grayScaleWthBorder = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat thresholded = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat eroded = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat dilated = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat hierarchy = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		
		cvtColor(origin, grayScale, COLOR_BGR2GRAY);
		if (debugPrepare) imwrite("grayscale.png", grayScale);
		
		threshold(grayScale, grayScaleHist, 128, 255, THRESH_OTSU);
		
		erode(grayScaleHist, grayScaleHist, getStructuringElement(CV_SHAPE_ELLIPSE, new Size(2 * 1 + 1, 2 * 1 + 1),
                new Point(1, 1)));
		
		//equalizeHist(grayScale, grayScaleHist);
		if (debugPrepare) imwrite("dbg/grayscalehist.png", grayScaleHist);
		//L’adaptation au changement plus que le suivi d’un plan
		
		inRange(grayScaleHist, new Mat(50.), new Mat(100.), inRange);
		if (debugPrepare) imwrite("dbg/inRange.png", inRange);
		
		
		copyMakeBorder(grayScaleHist, grayScaleWthBorder, 8, 8, 8, 8, BORDER_REPLICATE);
		if (debugPrepare) imwrite("dbg/grayScaleWthBorder.png", grayScaleWthBorder);
		
		threshold(grayScaleWthBorder, thresholded, 128, 255, THRESH_BINARY_INV | THRESH_OTSU);
		if (debugPrepare) imwrite("dbg/thresholded.png", thresholded);
		
		
		Mat preparedForContours = thresholded; 
		int kernelSize = 0;
		/*
		var elementType = CV_SHAPE_ELLIPSE;
        Mat element = getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));
        erode(thresholded, eroded, element);
        if (debugPrepare) imwrite("dbg/eroded.png", eroded);
		
        dilate(eroded, dilated, element);
        if (debugPrepare) imwrite("dbg/dilated.png", dilated);
        preparedForContours = dilated;
        */
        
		MatVector contours = new MatVector();
		findContours(preparedForContours, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		
		
		BufferedImage boxedDbg = ImageIO.read(new File(fileName));
		
		
		List<Rect> boxes = new ArrayList<>();
		for (int idx = 0; idx < contours.size(); idx ++) {
			Mat p = contours.get(idx);
			Rect rect = boundingRect(p);
			if (rect.area() < 32) continue;
			boxes.add(rect);
			//System.out.println("[] " + rect.x() + "-" +(rect.x() + rect.width()));
		}
		
		if (boxes.size() == 2) {
			List<Rect> tmp = new ArrayList<>();
			for (int idx = 0; idx < boxes.size(); idx ++) {
				Rect rect = boxes.get(idx);
				//System.out.println("split " + rect.x() + "-" +(rect.x() + rect.width())  + "  " + ((0.001 + rect.width()) / rect.height()));
				if ((0.001 + rect.width()) / rect.height() > 1.2) {
					Rect a = new Rect(rect.x(), rect.y(), rect.width()/2+1, rect.height());
					Rect b = new Rect(rect.x()+rect.width()/2, rect.y(), rect.width() - rect.width()/2+1, rect.height());
					tmp.add(a);
					tmp.add(b);
				}
				else {
					tmp.add(rect);
				}
			}
			boxes = tmp;
		}
		if (boxes.size() == 3) {
			List<Rect> tmp = new ArrayList<>();
			for (int idx = 0; idx < boxes.size(); idx ++) {
				Rect rect = boxes.get(idx);
				if ((0.001 + rect.width()) / rect.height() > 1.25) {
					Rect a = new Rect(rect.x(), rect.y(), rect.width()/2+1, rect.height());
					Rect b = new Rect(rect.x()+rect.width()/2, rect.y(), rect.width() - rect.width()/2+1, rect.height());
					tmp.add(a);
					tmp.add(b);
					//System.out.println("split " + rect.x() + "-" +(rect.x() + rect.width())  + "  " + ((0.001 + rect.width()) / rect.height()));
				}
				else {
					tmp.add(rect);
				}
			}
			boxes = tmp;
		}
		
		if (boxes.size() >= 5) {
			//search to merge 2 collinding boxes

			int idx_1 = 0;
			int idx_2 = 0;
			for (int idx1 = 0; idx1 < boxes.size(); idx1 ++) {
				Rect rect1 = boxes.get(idx1);
				for (int idx2 = 0; idx2 < boxes.size(); idx2 ++) {
					if (idx1 == idx2) continue;
					Rect rect2 = boxes.get(idx2);
					
					if (rect1.x() <= rect2.x() + rect2.width() +1
							&& rect1.x() >= rect2.x()) {
						idx_1 = idx1;
						idx_2 = idx2;
					}
				}
			}
			List<Rect> tmp = new ArrayList<>();
			for (int idx = 0; idx < boxes.size(); idx ++) {
				if (idx != idx_1 && idx != idx_2) {
					Rect rect = boxes.get(idx);
					tmp.add(rect);
				}
			}
			
			Rect rect1 = boxes.get(idx_1);
			Rect rect2 = boxes.get(idx_2);
			
			int mx = Math.min(rect1.x(), rect2.x());
			int my = Math.min(rect1.y(), rect2.y());
			int mw = Math.max(rect1.x()+rect1.width(), rect2.x()+rect2.width()) - mx;
			int mh = Math.max(rect1.y()+rect1.height(), rect2.y()+rect2.height()) - my;
			Rect merged= new Rect(mx, my,mw, mh);
			tmp.add(merged);
			
			boxes = tmp;
		}
		
		Map<Integer, GuessResult> guessed = new HashMap<>();
		for (int idx = 0; idx < boxes.size(); idx ++) {
			Rect rect = boxes.get(idx);
			//System.out.println(rect.x() + " " + rect.y()+ " " + rect.width() + " " + rect.height());
			//BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(Math.max(rect.x()-8, 0), Math.max(0, rect.y()-8), Math.min(original_width - Math.max(rect.x()-8, 0), rect.width()),
			//		Math.min(original_height - Math.max(rect.y()-8, 0), rect.height()));
			int x = Math.max(rect.x()-8 -1 -kernelSize, 0);
			int y = Math.max(0, rect.y()-8 -1 -kernelSize);
			int w = Math.min(original_width - Math.max(rect.x()-8 -1-kernelSize, 0), rect.width()+2+2*kernelSize);
			int h = Math.min(original_height - Math.max(rect.y()-8-1-kernelSize, 0), rect.height()+2+2*kernelSize);
			if (w <= 0 || h <= 0) continue;
			
			BufferedImage output;
			{
				BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(
						x, 
						y, 
						w,
						h);
				if (debugPrepare) ImageIO.write(cropped, "PNG", new File("dbg/cropped_"+idx+".png"));
				
				double dw = ((double)width)/w;
				double dh = ((double)height)/h;
				double d = Math.min(dw, dh);
				
				int fw = (int) (w * d +0.5);
				int fh = (int) (h * d +0.5);
				
				BufferedImageOp resampler0 = new ResampleOp(fw, fh, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
				BufferedImage sampled = resampler0.filter(cropped, null);

				BufferedImage enlarged = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
				Graphics g = enlarged.getGraphics();
				g.setColor(Color.white);
				g.fillRect(0, 0, width, height);
				g.drawImage(sampled, (width-fw)/2, (height-fh)/2, null);
				output = enlarged;
			}
			
			RescaleOp rescaleOp = new RescaleOp(1.2f, 15, null);
			rescaleOp.filter(output, output);
			
			output = invert(output);
			
			if (debugPrepare) ImageIO.write(output, "PNG", new File("dbg/output_"+idx+".png"));
			
			//if (debugPrepare) 
			{
				Graphics graphics = boxedDbg.getGraphics();
				graphics.setColor(Color.red);
				graphics.drawRect(Math.max(rect.x()-8-kernelSize, 0), Math.max(0, rect.y()-8-kernelSize), Math.min(original_width - Math.max(rect.x()-8-kernelSize, 0), rect.width()+2*kernelSize),
						Math.min(original_height - Math.max(rect.y()-8-kernelSize, 0), rect.height()+2*kernelSize));	
			}
			
			GuessResult g = net2 !=  null ? guess(width, height, net2, output) : new GuessResult("", 0.);
			if (g.confident > 0.2) guessed.put(rect.x(), g);
			
			if (debugNet) {
				System.out.println(idx + ":" + g.guess + " (" + g.confident + ")");
			}
			
		}
		
		String r = guessed.entrySet().stream().sorted(Comparator.comparing(e -> e.getKey())).map(e -> e.getValue().guess).collect(Collectors.joining());
		double c = guessed.entrySet().stream().sorted(Comparator.comparing(e -> e.getKey())).mapToDouble(e -> e.getValue().confident).reduce(1, (a,b) -> a*b);

		//if (debugPrepare) 
		{
			ImageIO.write(boxedDbg, "PNG", new File("dbg/boxed"+new File(fileName).getName().split("\\.")[0]+"_"+r+".png"));
			
		}
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
