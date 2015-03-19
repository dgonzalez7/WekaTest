import java.text.DecimalFormat;
import java.util.Random;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;

public class NNTrainAndTest {
	
	private static DecimalFormat df = new DecimalFormat("0.0000");

	public static void main(String[] args) throws Exception 
	{
		// Train and Test Sets		
		System.out.println("*** Train & Test ***");
		
		// DataSource dsTrain = new DataSource("dataset/weather.numeric.arff");
		DataSource dsTrain = new DataSource("dataset/titanic.randomized.top75pct.arff");
		Instances train = dsTrain.getDataSet(); 
		// DataSource dsTest = new DataSource("dataset/weather.numeric.arff");
		DataSource dsTest = new DataSource("dataset/titanic.randomized.bottom25pct.arff");
		Instances test = dsTest.getDataSet(); 
		
		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		
		String[] options = new String[2];
        options[0] = "-H";
        options[1] = "4,5";
        // options[1] = "4";
        // System.out.println("Hidden Layer String: " + options[1]);
		
		MultilayerPerceptron cls = new MultilayerPerceptron();
        cls.setOptions(options);
        cls.setAutoBuild(true);
        cls.setSeed(0);
        // cls.setHiddenLayers(Integer.toString(getHiddenLayers()));
        // cls.setNominalToBinaryFilter(false);
        cls.setNormalizeAttributes(true);
        cls.setNormalizeNumericClass(true);
        cls.setDecay(false);
        cls.setLearningRate(0.3);
        cls.setMomentum(0.2);
        cls.setTrainingTime(500);
        cls.setNominalToBinaryFilter(false);
        // cls.setGUI(true);        
        cls.buildClassifier(train);
		
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(cls, test);
		System.out.println("Filter: " + cls.getNominalToBinaryFilter() + "\n");
		System.out.println(eval.toSummaryString("Results\n", false));		
		System.out.println("Test correct = " + eval.correct() +
				" (" + df.format(eval.correct()/test.numInstances()*100.0) + "%)");
		System.out.println("Test incorrect = " + eval.incorrect() +
				" (" + df.format(eval.incorrect()/test.numInstances()*100.0) + "%)");
		
		// Cross-validation of full set		
		System.out.println("\n*** Cross-Validation on Full Dataset ***");
		
		// DataSource dsTrain = new DataSource("dataset/weather.numeric.arff");
		DataSource dsFull = new DataSource("dataset/titanic.randomized.arff");
		Instances full = dsFull.getDataSet(); 
		
		full.setClassIndex(full.numAttributes() - 1);
		
		String[] fullOptions = new String[2];
        fullOptions[0] = "-H";
        fullOptions[1] = "4,5";
        // fullOptions[1] = "4";       
        // System.out.println("Hidden Layer String: " + options[1]);
		
		MultilayerPerceptron clsFull = new MultilayerPerceptron();
		clsFull.setAutoBuild(true);
		clsFull.setSeed(0);
        // cls.setHiddenLayers(Integer.toString(getHiddenLayers()));
		clsFull.setNominalToBinaryFilter(false);
        clsFull.setNormalizeAttributes(true);
        clsFull.setNormalizeNumericClass(true);
        clsFull.setDecay(false);
		clsFull.setLearningRate(0.3);
		clsFull.setMomentum(0.2);
		clsFull.setTrainingTime(500);
		clsFull.setOptions(fullOptions);
		// clsFull.setGUI(true);
		clsFull.buildClassifier(full);
		
		Evaluation evalFull = new Evaluation(full);
		evalFull.crossValidateModel(clsFull, full, 10, new Random(1));
		
		System.out.println(evalFull.toSummaryString("Results\n", false));		
		System.out.println("Test correct = " + evalFull.correct() +
				" (" + df.format(evalFull.correct()/full.numInstances()*100.0) + "%)");
		System.out.println("Test incorrect = " + evalFull.incorrect() +
				" (" + df.format(evalFull.incorrect()/full.numInstances()*100.0) + "%)");
	}
}