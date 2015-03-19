import java.text.DecimalFormat;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NeuralNetworkRun 
{
	// Set-up variables 		
	private static boolean flagNominalToBinaryFilter = false;
	private static double learningRate = 0.3;
	private static double momentumRate = 0.2;
	private static int trainingTime = 500;
	private static int minNeuronCount = 4;
	private static int maxNeuronCount = 10;
	private static int minNeuronCount2 = 4;
	private static int maxNeuronCount2 = 10;
		
	private static DecimalFormat df = new DecimalFormat("0.0000");
	
	public NeuralNetworkRun(String fullTrainingDataset, String trainingDataset, String testingDataset) throws Exception  
	{
		// Train & Test Sets
		for (int i = minNeuronCount; i <= maxNeuronCount; i++)
		{
			trainAndTest(trainingDataset, testingDataset, i, 0);
		}
		for (int i = minNeuronCount; i <= maxNeuronCount; i++)
		{
			for (int j = minNeuronCount2; j <= maxNeuronCount2; j++)
			{
				trainAndTest(trainingDataset, testingDataset, i, j);
			}
		}
		
		// Cross Validation
		for (int i = minNeuronCount; i <= maxNeuronCount; i++)
		{
			crossValidation(fullTrainingDataset, i, 0);
		}
		for (int i = minNeuronCount; i <= maxNeuronCount; i++)
		{
			for (int j = minNeuronCount2; j <= maxNeuronCount2; j++)
			{
				crossValidation(fullTrainingDataset, i, j);
			}
		}
	}
	
	public void trainAndTest(String trainingDataset, String testingDataset, int layer1, int layer2) throws Exception
	{
		String neuronStructure;
		// = Integer.toString(i);
		if (layer2 == 0)
		{
			neuronStructure = Integer.toString(layer1);
		}
		else
		{
			neuronStructure = Integer.toString(layer1) + "," + Integer.toString(layer2);
		}
		
		// String neuronStructure = Integer.toString(i);

		// Train and Test Sets		
		System.out.println("*** Train & Test ***");

		DataSource dsTrain = new DataSource(trainingDataset);
		Instances train = dsTrain.getDataSet(); 
		DataSource dsTest = new DataSource(testingDataset);
		Instances test = dsTest.getDataSet(); 

		train.setClassIndex(train.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);

		String[] options = new String[2];
		options[0] = "-H";
		// options[1] = "4,5";
		options[1] = neuronStructure;
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
		cls.setLearningRate(learningRate);
		cls.setMomentum(momentumRate);
		cls.setTrainingTime(trainingTime);
		cls.setNominalToBinaryFilter(flagNominalToBinaryFilter);
		// cls.setGUI(true);        
		cls.buildClassifier(train);

		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(cls, test);

		// Print configuration
		printConfig(cls);

		// Print results
		printResults(eval, test.numInstances());
	}	
	
	public void crossValidation(String fullTrainingDataset, int layer1, int layer2) throws Exception
	{
		String neuronStructure;
		// = Integer.toString(i);
		if (layer2 == 0)
		{
			neuronStructure = Integer.toString(layer1);
		}
		else
		{
			neuronStructure = Integer.toString(layer1) + "," + Integer.toString(layer2);
		}
		
		// Cross-validation of full set		
		System.out.println("*** Cross-Validation on Full Dataset ***");

		// DataSource dsTrain = new DataSource("dataset/weather.numeric.arff");
		DataSource dsFull = new DataSource(fullTrainingDataset);
		Instances full = dsFull.getDataSet(); 

		full.setClassIndex(full.numAttributes() - 1);

		String[] fullOptions = new String[2];
		fullOptions[0] = "-H";
		// fullOptions[1] = "4,5";
		fullOptions[1] = neuronStructure;       
		// System.out.println("Hidden Layer String: " + options[1]);

		MultilayerPerceptron clsFull = new MultilayerPerceptron();
		clsFull.setAutoBuild(true);
		clsFull.setSeed(0);
		// cls.setHiddenLayers(Integer.toString(getHiddenLayers()));
		clsFull.setNormalizeAttributes(true);
		clsFull.setNormalizeNumericClass(true);
		clsFull.setDecay(false);
		clsFull.setLearningRate(learningRate);
		clsFull.setMomentum(momentumRate);
		clsFull.setTrainingTime(trainingTime);
		clsFull.setOptions(fullOptions);
		clsFull.setNominalToBinaryFilter(flagNominalToBinaryFilter);
		// clsFull.setGUI(true);
		clsFull.buildClassifier(full);

		Evaluation evalFull = new Evaluation(full);
		evalFull.crossValidateModel(clsFull, full, 10, new Random(1));

		// Print configuration
		printConfig(clsFull);
		// Print results
		printResults(evalFull, full.numInstances());
	}
	
	public void printConfig(MultilayerPerceptron mlp)
	{
		// Print configuration
		System.out.println("*** Configuration ***");
		System.out.println("NominalToBinaryFilter: " + mlp.getNominalToBinaryFilter());
		System.out.println("NormalizeNumericClass: " + mlp.getNormalizeNumericClass());
		System.out.println("NormalizeAttributes: " + mlp.getNormalizeAttributes());
		System.out.println("Learning Rate: " + mlp.getLearningRate());
		System.out.println("Momentum Rate: " + mlp.getMomentum());
		System.out.println("Hidden Layers: " + mlp.getHiddenLayers());
		System.out.println("*********************\n");
	}

	public void printResults(Evaluation eval, int numInstances)
	{
		// Print results
		System.out.println(eval.toSummaryString("Results\n", false));		
		System.out.println("Test correct = " + eval.correct() +
				" (" + df.format(eval.correct()/numInstances*100.0) + "%)");
		System.out.println("Test incorrect = " + eval.incorrect() +
				" (" + df.format(eval.incorrect()/numInstances*100.0) + "%)\n");
	}
}
