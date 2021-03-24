package crackTheCaptcha;

import static org.bytedeco.opencv.global.opencv_core.BORDER_REPLICATE;
import static org.bytedeco.opencv.global.opencv_core.add;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;
import static org.bytedeco.opencv.global.opencv_core.extractChannel;
import static org.bytedeco.opencv.global.opencv_core.inRange;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_NONE;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_MOP_OPEN;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_SHAPE_ELLIPSE;
import static org.bytedeco.opencv.global.opencv_imgproc.FILLED;
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_EXTERNAL;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_OTSU;
import static org.bytedeco.opencv.global.opencv_imgproc.boundingRect;
import static org.bytedeco.opencv.global.opencv_imgproc.dilate;
import static org.bytedeco.opencv.global.opencv_imgproc.drawContours;
import static org.bytedeco.opencv.global.opencv_imgproc.findContours;
import static org.bytedeco.opencv.global.opencv_imgproc.getStructuringElement;
import static org.bytedeco.opencv.global.opencv_imgproc.morphologyEx;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;
import static org.bytedeco.opencv.global.opencv_photo.INPAINT_NS;
import static org.bytedeco.opencv.global.opencv_photo.inpaint;

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
import java.text.DecimalFormat;
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
import org.bytedeco.opencv.opencv_core.Scalar;
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
 * https://gist.github.com/christianroman/5679049
 * https://mathieularose.com/decoding-captchas/
 * 
 * @author gaeta
 *
 */
public class TrainingPrepX5 {

	static boolean debugNet = false;
	static boolean debugPrepare = false;
	static int width = 28;
	static int height = 28;
	static BufferedImageOp resampler = new ResampleOp(width, height, ResampleOp.FILTER_LANCZOS); // A good default filter, see class documentation for more info
	static ImageFilter filter = new GrayFilter();
	static DecimalFormat df2 = new DecimalFormat("0.##");
	static DecimalFormat df2_ = new DecimalFormat("##");
	
	public static void main(String[] args) throws IOException {
		int width = 28;
		int height = 28;
		
		debugPrepare = true;
		debugNet = true;
		
		if (debugPrepare) {
			if (new File("dbg").exists()) {
				File[] listFiles = new File("dbg").listFiles();
				for (File f : listFiles) {
					f.delete();
				}
			}
		}
		
		
		MultiLayerNetwork net2 = MultiLayerNetwork.load(new File("models/emnist1/bestModel.bin"), false);
		//MultiLayerNetwork net2 = null;//MultiLayerNetwork.load(new File("emnist_model.bin"), false);
		//GuessResult analyse = analyse(width, height, net2, filter, resampler, "test/ACWU.png");
		
		if (net2 != null) {
			System.out.println(net2.summary());
			System.out.println();
		}
		GuessResult analyse = analyse(width, height, net2, "3bIHDds.png");
		//GuessResult analyse = analyse(width, height, net2, filter, resampler, "4pVrm.jpg");
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
		
		
		File evalDirectory = new File("test");

		int correct = 0;
		int all = 0;
		for (File f : evalDirectory.listFiles()) {
			String reference = f.getName().substring(0, 4);
			
			GuessResult analyse = analyse(width, height, net2, f.getAbsolutePath());
			if (reference.equals(analyse.guess)) {
				correct++;
			}
			all++;
			
			System.out.println(analyse.guess + ";" + reference+";"+ analyse.confident);
		}
		
		System.out.println("Captcha " + correct  + "/" + all);
		
	}

	private static GuessResult analyse(int width, int height, MultiLayerNetwork net2, String fileName) throws IOException {
		BufferedImage read = ImageIO.read(new File(fileName));
		if (debugPrepare) ImageIO.write(read, "PNG", new File("dbg/"+new File(fileName).getName().split("\\.")[0]+".png"));
		
		int original_width = read.getWidth();
		int original_height = read.getHeight();
		
		ImageProducer producer = new FilteredImageSource(read.getSource(), filter);  
		Image mage = Toolkit.getDefaultToolkit().createImage(producer);
		
		Mat origin  = imread(fileName);
		Mat byCol = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat hierarchyOrigin = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat grayScale = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat tmpo = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat grayScaleHist = new Mat(origin.size().width(), origin.size().height(), COLOR_BGR2GRAY);
		Mat grayScaleWthBorder = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat inRange = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat morphed = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat thresholded = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat tmp = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat tmp2 = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat eroded = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat dilated = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		Mat hierarchy = new Mat(origin.size().width() + 16, origin.size().height() + 16, COLOR_BGR2GRAY);
		
		
		extractChannel(origin, grayScale, 1);
		
		//cvtColor(byCol, grayScale, COLOR_BGR2GRAY);
		
				
		if (debugPrepare) imwrite("dbg/origin.png", origin);
		if (debugPrepare) imwrite("dbg/grayscale.png", grayScale);
		
		
		{
			threshold(grayScale, tmpo, 128, 255, THRESH_OTSU | THRESH_BINARY_INV);
			if (debugPrepare) imwrite("dbg/origin_threshold.png", tmpo);

			MatVector contours = new MatVector();
			findContours(tmpo, contours, hierarchyOrigin, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			
			List<Rect> boxes = new ArrayList<>();
			for (int idx = 0; idx < contours.size(); idx ++) {
				Mat p = contours.get(idx);
				
				Mat cl = origin.clone();
				//fillPoly(cl, contours, new Scalar(255 ,0 , 0, 255));
				//drawContours(cl, contours, idx, new Scalar(0 ,0 , 255, 255));
				drawContours(cl, contours, idx, new Scalar(0 ,0 , 255, 255), FILLED, 8, hierarchyOrigin, 1, new Point(0,0));
				
				//if (debugPrepare) imwrite("dbg/origin_bruit_"+idx+"_.png", cl);
				Rect rect = boundingRect(p);
				if (rect.area() < 32) {
					drawContours(grayScale, contours, idx, new Scalar(255), FILLED, 8, hierarchyOrigin, 1, new Point(0,0));
					
				}
			}
			
		}
		
		
		//threshold(grayScale, grayScaleHist, 128, 255, THRESH_OTSU);
		//equalizeHist(grayScale, grayScaleHist);
		//if (debugPrepare) imwrite("dbg/grayscalehist.png", grayScaleHist);
		
		
		copyMakeBorder(grayScale, grayScaleWthBorder, 8, 8, 8, 8, BORDER_REPLICATE);
		if (debugPrepare) imwrite("dbg/grayScaleWthBorder.png", grayScaleWthBorder);
		
		//for (int i = 0; i < 255; i += 4)
			
		int sx = 112;
		int dx = 8;
		
		//for (dx = 4; dx < 32; dx += 4)
		//for (sx = 0; sx < 255 - dx; sx += dx) 
		{

			var sb = new Mat(new double[]{sx});
			var su = new Mat(new double[]{sx + dx});

			inRange(grayScaleWthBorder, sb, su, inRange);
			if (debugPrepare) imwrite("dbg/inrange_"+dx+"_"+sx+".png", inRange);
			if (debugPrepare) imwrite("dbg/inrange_"+dx+"_"+sx+".png", inRange);
			
			add(grayScaleWthBorder, inRange, tmp);
			if (debugPrepare) imwrite("dbg/inrangeAdd_"+dx+"_"+sx+".png", inRange);

			{
				int kernelSizeDilate = 2;//0-2
				Mat elementDilate = getStructuringElement(CV_SHAPE_ELLIPSE, new Size(2 * kernelSizeDilate+ 1, 2 * kernelSizeDilate + 1),
						new Point(kernelSizeDilate, kernelSizeDilate));
				dilate(inRange, inRange, elementDilate);
			}
			
			inpaint(grayScaleWthBorder, inRange, tmp, 4, INPAINT_NS);
			if (debugPrepare) imwrite("dbg/inrangeInPaint_"+dx+"_"+sx+".png", tmp);
			
			
			int kernelSize = 0;//0-2
			var elementType = CV_SHAPE_ELLIPSE;
			/*
	        Mat element = getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
	                new Point(kernelSize, kernelSize));
	        erode(tmp, eroded, element);
	        if (debugPrepare) imwrite("dbg/eroded"+dx+"_"+sx+".png", eroded);
			*/
			int kernelSizeDilate = 0;//0-2
	        Mat elementDilate = getStructuringElement(elementType, new Size(2 * kernelSizeDilate+ 1, 2 * kernelSizeDilate + 1),
	        		new Point(kernelSizeDilate, kernelSizeDilate));
	        dilate(tmp, tmp2, elementDilate);
	        if (debugPrepare) imwrite("dbg/dilated"+dx+"_"+sx+".png", tmp2);
	        tmp2 = tmp;
	        
			threshold(tmp2, tmp, 128, 255, THRESH_OTSU | THRESH_BINARY_INV);
			if (debugPrepare) imwrite("dbg/inrangeThresh_"+dx+"_"+sx+".png", tmp);
	        
	    	MatVector contours = new MatVector();
			findContours(tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			
			
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
			
			if (dx ==8 && sx == 112) {
				System.out.println();
			}
			
			Map<Integer, GuessResult> guessed = new HashMap<>();
			String explain ="";
			for (int idx = 0; idx < boxes.size(); idx ++) {
				Rect rect = boxes.get(idx);
				
				if (rect.area() < 64) continue;
				
				if (debugPrepare) {
					Graphics graphics = boxedDbg.getGraphics();
					graphics.setColor(Color.red);
					graphics.drawRect(Math.max(rect.x()-8-kernelSize, 0), Math.max(0, rect.y()-8-kernelSize), Math.min(original_width - Math.max(rect.x()-8-kernelSize, 0), rect.width()+2*kernelSize),
							Math.min(original_height - Math.max(rect.y()-8-kernelSize, 0), rect.height()+2*kernelSize));	
				}
				
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
				
				
				if (debugPrepare) ImageIO.write(output, "PNG", new File("dbg/zoutput_"+dx+"_"+sx+"_"+idx+".png"));
				
				
				GuessResult g = net2 !=  null ? guess(width, height, net2, invert(output)) : new GuessResult("", 0.);
				
				explain+="("+idx+":"+g.guess+":"+df2.format(g.confident)+":"+rect.area()+")";

				if (rect.area() < 64) continue;
				
				if (g.confident > 0.2) {
					guessed.put(rect.x(), g);
					
				}
				if (debugPrepare) ImageIO.write(output, "PNG", new File("dbg/zoutput_guessed_"+g.guess+"_"+df2.format(g.confident).substring(2)+"_"+dx+"_"+sx+"_"+idx+".png"));
				
			}
			if (debugPrepare) {
				ImageIO.write(boxedDbg, "PNG", new File("dbg/output_boxed_"+dx+"_"+sx+".png"));
				
				String r = guessed.entrySet().stream().sorted(Comparator.comparing(e -> e.getKey())).map(e -> e.getValue().guess).collect(Collectors.joining());
				double c = guessed.entrySet().stream().sorted(Comparator.comparing(e -> e.getKey())).mapToDouble(e -> e.getValue().confident).reduce(1, (a,b) -> a*b);
				
				if (c > 0.001 && r.length()>3 && r.length() < 12) {
					System.out.println(r + " (" + c + ")  " +dx+"_"+sx + " " + explain);
				}
			}
			
			
			
		}
		
		
		{
			int i = 32;
			threshold(grayScaleWthBorder, thresholded, i*4, 255, THRESH_BINARY_INV);
			if (debugPrepare) imwrite("dbg/thresholded_"+i+".png", thresholded);
			
			add(grayScaleWthBorder, thresholded, tmp);
			if (debugPrepare) imwrite("dbg/thresholded_add_"+i+".png", tmp);

			threshold(tmp, thresholded, 128, 255, THRESH_OTSU);
			
			int kernelSizeDilate = 2;
			Mat elementDilate = getStructuringElement(CV_SHAPE_ELLIPSE, new Size(2 * kernelSizeDilate+ 1, 2 * kernelSizeDilate + 1),
	        		new Point(kernelSizeDilate, kernelSizeDilate));
	        dilate(thresholded, dilated, elementDilate);
	        if (debugPrepare) imwrite("dbg/thresholded_dilated"+i+".png", dilated);
			
			
		}
		
		{
			int kernelSizeOpen = 3;//0-2
			Mat elementMorphOpen = getStructuringElement(CV_SHAPE_ELLIPSE, new Size(2 * kernelSizeOpen + 1, 2 * kernelSizeOpen + 1),
					new Point(kernelSizeOpen, kernelSizeOpen));
			morphologyEx(dilated, thresholded, CV_MOP_OPEN, elementMorphOpen);
			//morphologyEx(grayScaleWthBorder, morphed, CV_MOP_OPEN, elementMorphOpen);
			if (debugPrepare) imwrite("dbg/morphed.png", thresholded);
			dilated = thresholded;
		}
		
		int kernelSize = 2;//0-2
		/*
		int kernelSizeDilate = 2;//0-2
		var elementType = CV_SHAPE_ELLIPSE;
        Mat element = getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));
        erode(thresholded, eroded, element);
        if (debugPrepare) imwrite("dbg/eroded.png", eroded);
		
        Mat elementDilate = getStructuringElement(elementType, new Size(2 * kernelSizeDilate+ 1, 2 * kernelSizeDilate + 1),
        		new Point(kernelSize, kernelSize));
        dilate(eroded, dilated, element);
        if (debugPrepare) imwrite("dbg/dilated.png", dilated);
        */
        
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
			
			if (rect.area() < 64) continue;
			
			//System.out.println(rect.x() + " " + rect.y()+ " " + rect.width() + " " + rect.height());
			//BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(Math.max(rect.x()-8, 0), Math.max(0, rect.y()-8), Math.min(original_width - Math.max(rect.x()-8, 0), rect.width()),
			//		Math.min(original_height - Math.max(rect.y()-8, 0), rect.height()));
			BufferedImage cropped = new BufferedImageFactory(mage).getBufferedImage().getSubimage(
					Math.max(rect.x()-8 -1 -kernelSize, 0), 
					Math.max(0, rect.y()-8 -1 -kernelSize), 
					Math.min(original_width - Math.max(rect.x()-8 -1-kernelSize, 0), rect.width()+2+2*kernelSize),
					Math.min(original_height - Math.max(rect.y()-8-1-kernelSize, 0), rect.height()+2+2*kernelSize));
			if (debugPrepare) ImageIO.write(cropped, "PNG", new File("dbg/cropped_"+idx+".png"));
			
			BufferedImage output = resampler.filter(cropped, null);
			if (debugPrepare) ImageIO.write(output, "PNG", new File("dbg/output_"+idx+".png"));
			
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
			ImageIO.write(boxedDbg, "PNG", new File("dbg/output_boxed.png"));
			
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
