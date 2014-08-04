import weka.core.*; 
import weka.core.converters.ConverterUtils.DataSource;

public class Test {

	public static void main(String[] args) throws Exception 
	{
		System.out.println("Weka loaded!!!");
		LoadData ld = new LoadData();
		FilterAttribute fa = new FilterAttribute(); 
		SelectAttributes sa = new SelectAttributes();
		TrainClassifier tc = new TrainClassifier();
		// MyClassifier mc = MyClassifier();
	}

}
