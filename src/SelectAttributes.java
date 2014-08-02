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
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


public class SelectAttributes {

	SelectAttributes() throws Exception
	{
		System.out.println("\nSelect Attributes called\n");
		
		Instances myDataset = loadDataset("dataset/titanic.arff");

		// Instances myDataset = createDataset();
		
		selectExample(myDataset);
		infoGainExample(myDataset);
		pcaExample(myDataset);
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
	
	
	public static void selectExample(Instances dataset) throws Exception
	{
		System.out.println("selectExample called");
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(dataset);
		Instances newDataset = Filter.useFilter(dataset, filter);
		
		// System.out.println("\nOriginal Data");
		// System.out.println(dataset.toString());
		// System.out.println("\nFiltered Data");
		// System.out.println(newDataset.toString());
		System.out.println(newDataset.numAttributes());
	}
	
	public static void infoGainExample(Instances dataset) throws Exception
	{
		System.out.println("infoGainExample called");
		weka.attributeSelection.AttributeSelection attSelect = new weka.attributeSelection.AttributeSelection();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();

		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		attSelect.SelectAttributes(dataset);
		
		int[] indices = attSelect.selectedAttributes();
		
		System.out.println("\nFiltered Data");
		System.out.println(attSelect.toResultsString());
		System.out.println(Utils.arrayToString(indices));
	}
	
	public static void pcaExample(Instances dataset) throws Exception
	{
		System.out.println("pcaExample called");
		AttributeSelection filter = new AttributeSelection();
		PrincipalComponents eval = new PrincipalComponents();
		Ranker search = new Ranker();

		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(dataset);
		// generate new data
		Instances newDataset = Filter.useFilter(dataset, filter);
		
		System.out.println("\nPCA Filtered Data");
		System.out.println(newDataset);
	}
}
