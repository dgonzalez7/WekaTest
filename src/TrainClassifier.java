import java.util.Random;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;


public class TrainClassifier {

	TrainClassifier() throws Exception
	{
		System.out.println("\nTrain Classifier called\n");
		
		Instances myDataset = loadDataset("dataset/titanic.arff");

		// Instances myDataset = createDataset();
		
		buildClassifier(myDataset);
		//infoGainExample(myDataset);
		//pcaExample(myDataset);
		//classifierSpecificExample(myDataset);
	}
	
	public static Instances loadDataset(String dataset) throws Exception
	{
		System.out.println("loadData called");
		DataSource source = new DataSource(dataset);
		
		// Load the data
		Instances data = source.getDataSet();
		
		System.out.println(data.numInstances() + " instances loaded.");
		// System.out.println(data.toString());
		
		return data;
	}
	
	
	public static void buildClassifier(Instances dataset) throws Exception
	{
		System.out.println("buildClassifier called");
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		// Decision Tree
		String[] options = new String[1];
		options[0] = "-U";
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(dataset);
		System.out.println("J48 Decision Tree");
		System.out.println(tree);

		// Support vector machine
		SMO svm = new SMO();
		svm.buildClassifier(dataset);
		System.out.println("Support Vector Machine");
		System.out.println(svm);
	}
	
}
