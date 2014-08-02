import java.text.ParseException;
import weka.core.*; 
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Discretize;
import java.io.File;
import java.io.IOException;

public class FilterAttribute {

	FilterAttribute() throws Exception
	{
		System.out.println("\nFilter Attribte called\n");
		
		Instances myDataset = loadDataset("dataset/titanic.arff");

		// Instances myDataset = createDataset();
		
		filterDataset(myDataset);
		
		attributeDiscretization(myDataset);
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
	
	public void filterDataset(Instances dataset) throws Exception 
	{
		String[] opts = new String[] {"-R", "2"};
		Remove remove = new Remove();
		remove.setOptions(opts);
		remove.setInputFormat(dataset);
		Instances newDataset = Filter.useFilter(dataset, remove);
		
		System.out.println("\nOriginal Data");
		System.out.println(dataset.toString());
		System.out.println("\nFiltered Data");
		System.out.println(newDataset.toString());
	}
	
	public void attributeDiscretization(Instances dataset) throws Exception 
	{
		String[] opts = new String[4];
		opts[0] = "-B";
		opts[1] = "2";
		opts[2] = "-R";
		opts[3] = "first-last";
		
		Discretize discretize = new Discretize();
		discretize.setOptions(opts);
		discretize.setInputFormat(dataset);
		Instances newDataset = Filter.useFilter(dataset, discretize);
		
		System.out.println("\nOriginal Data");
		System.out.println(dataset.toString());
		System.out.println("\nDiscretized Data");
		System.out.println(newDataset.toString());
	}

}
