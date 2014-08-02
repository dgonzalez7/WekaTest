import java.text.ParseException;
import weka.core.*; 
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.converters.ArffSaver;
import java.io.File;
import java.io.IOException;

public class LoadData {

	LoadData() throws Exception
	{
		System.out.println("More");
		
		loadDataset("dataset/titanic.arff");

		Instances myDataset = createDataset();
		
		saveDataset(myDataset, true);
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
	
	public static Instances createDataset() throws ParseException
	{
		// 1 Set up attributes
		FastVector attributes = new FastVector();
		
		// add nominal attribute
		FastVector catVals = new FastVector(3);
		catVals.addElement("sports");
		catVals.addElement("finance");
		catVals.addElement("news");
		attributes.addElement(new Attribute("category (att1)", catVals));
		
		// add numeric attributes
		attributes.addElement(new Attribute("visits (att2)"));
		
		// add string attributes
		attributes.addElement(new Attribute("title (att3)", (FastVector) null));
		
		// add date attributes
		attributes.addElement(new Attribute("posted (att4)", "yyyy-MM-dd"));
		
		// 2 Create first instance
		Instances data = new Instances("Runtime dataset", attributes, 0);
		
		// add first instance 
		double vals[] = new double[data.numAttributes()];
		
		// nominal
		vals[0] = catVals.indexOf("sports");
		// numeric
		vals[1] = 8527.0;
		// string
		vals[2] = data.attribute(2).addStringValue("2012 Summer Olympics in London");
		// date
		vals[3] = data.attribute(3).parseDate("2012-07-27");
		data.add(new Instance(1.0, vals));
		
		// add second instance 
		vals = new double[data.numAttributes()];
		
		// nominal
		vals[0] = catVals.indexOf("finance");
		// numeric
		vals[1] = Instance.missingValue();
		// string
		vals[2] = data.attribute(2).addStringValue("Greek government debt crisis");
		// date
		vals[3] = data.attribute(3).parseDate("2012-02-01");
		data.add(new Instance(1.0, vals));
		
		System.out.println(data);
		
		return data;
	}
	
	public static void saveDataset(Instances dataset, boolean batchSave) throws IOException
	{
		ArffSaver saver = new ArffSaver();
		
		if (batchSave)
		{
			saver.setInstances(dataset);
			saver.setFile(new File("./data/test.arff"));
			saver.writeBatch();
		}
		else
		{
			saver.setRetrieval(ArffSaver.INCREMENTAL);
			saver.setInstances(dataset);
			saver.setFile(new File("./data/test.arff"));
			for (int i=0; i < dataset.numInstances(); i++)
			{
				saver.writeIncremental(dataset.instance(i));
			}
			saver.writeIncremental(null);
		}
	}
}
