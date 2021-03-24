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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
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
public class TrainingSet {

	static boolean debugNet = false;
	static boolean debugPrepare = false;
	
	public static void main(String[] args) throws IOException {

		var prep = new File("prep");
		var train = new File(prep, "train");
		if (!train.exists()) train.mkdirs();
		
		int width = 28;
		int height = 28;
		
		
		System.out.println("Load");

		ImageFilter filter = new GrayFilter();  
		BufferedImageOp resampler = new ResampleOp(width, height, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
		
		
		File evalDirectory = new File("train");

		for (File f : evalDirectory.listFiles()) {
			String reference = f.getName().substring(0, 4);
			analyse(width, height, filter, resampler, f.getAbsolutePath(), reference);
		}
		
		
	}
	
	static AtomicInteger counter = new AtomicInteger();

	private static void analyse(int width, int height, ImageFilter filter, BufferedImageOp resampler, String fileName, String reference) throws IOException {
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
		
		int kernelSize = 0;
		var elementType = CV_SHAPE_ELLIPSE;
        Mat element = getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));
        erode(thresholded, eroded, element);
        if (debugPrepare) imwrite("eroded.png", eroded);
		
        dilate(eroded, dilated, element);
        if (debugPrepare) imwrite("dilated.png", dilated);
        
		MatVector contours = new MatVector();
		findContours(dilated, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		
		
		//BufferedImage boxedDbg = debugPrepare ? ImageIO.read(new File(fileName)) : null;
		BufferedImage boxedDbg = ImageIO.read(new File(fileName));
		
		

		
		List<Rect> boxes = new ArrayList<>();
		for (int idx = 0; idx < contours.size(); idx ++) {
			Mat p = contours.get(idx);
			Rect rect = boundingRect(p);
			if (rect.area() < 32) continue;
			boxes.add(rect);
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
					System.out.println("split " + rect.x() + "-" +(rect.x() + rect.width())  + "  " + ((0.001 + rect.width()) / rect.height()));
				}
				else {
					tmp.add(rect);
				}
			}
			boxes = tmp;
		}
		
		if (boxes.size() == 5) {
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
		
		class Box {
			int x, y, w, h;

			public Box(int x, int y, int w, int h) {
				super();
				this.x = x;
				this.y = y;
				this.w = w;
				this.h = h;
			}
			
		}
		List<Box> boxesReady = new ArrayList<>();
		for (int idx = 0; idx < boxes.size(); idx ++) {
			Rect rect = boxes.get(idx);
			
			int x = Math.max(rect.x()-8 -1 -kernelSize, 0);
			int y = Math.max(0, rect.y()-8 -1 -kernelSize);
			int w = Math.min(original_width - Math.max(rect.x()-8 -1-kernelSize, 0), rect.width()+2+2*kernelSize);
			int h = Math.min(original_height - Math.max(rect.y()-8-1-kernelSize, 0), rect.height()+2+2*kernelSize);
			if (w <= 0 || h <= 0) continue;

			boxesReady.add(new Box(x,y,w,h));
		}
		
		Collections.sort(boxesReady, Comparator.comparing(b -> b.x));

		if (boxesReady.size() == 4) {
			for (int idx = 0; idx < boxesReady.size(); idx ++) {
				var rect = boxesReady.get(idx);
				//System.out.println(rect.x() + " " + rect.y()+ " " + rect.width() + " " + rect.height());
				//BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(Math.max(rect.x()-8, 0), Math.max(0, rect.y()-8), Math.min(original_width - Math.max(rect.x()-8, 0), rect.width()),
				//		Math.min(original_height - Math.max(rect.y()-8, 0), rect.height()));
				int x = rect.x;
				int y = rect.y;
				int w = rect.w;
				int h = rect.h;

				
				String letter = "" + reference.charAt(idx);
				
				BufferedImage output;
				{
					BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(
							x, 
							y, 
							w,
							h);
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
				

				if (!new File("prep/train/"+letter).exists()) new File("prep/train/"+letter).mkdirs();
				ImageIO.write(invert(output), "PNG", new File("prep/train/"+letter+"/"+reference+"_"+counter.incrementAndGet()+".png"));

				//if (debugPrepare) 
				{
					Graphics graphics = boxedDbg.getGraphics();
					graphics.setColor(Color.red);
					graphics.drawRect(x,y,w,h);	
				}
			}

			ImageIO.write(boxedDbg, "PNG", new File("dbg/box_"+reference+".png"));
			if (debugPrepare) {
			}
		}
		
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
