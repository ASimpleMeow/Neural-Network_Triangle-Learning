import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
/**
 * A neural network which uses the neuroph library
 * Learns the NAND function
 * 
 * @author Oleksandr Kononov
 * @version 18-02-2017
 *
 */
public class NeuralNetworkNand{
	
	public static void main(String[] args){
		// create new perceptron network
		NeuralNetwork neuralNetwork = new Perceptron(2, 1);
		// create training set
		DataSet trainingSet = new DataSet(2,1);
		
		//add training data to training set (logical NAND function)
		trainingSet.addRow(new DataSetRow(new double[] {0,0},new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {0,1}, new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {1,0}, new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {1,1}, new double[] {0}));
		//learn the training set
		neuralNetwork.learn(trainingSet);
		
		neuralNetwork.setInput(0,1);
		
		neuralNetwork.calculate();
		
		for(double val : neuralNetwork.getOutput())
			System.out.println(val);
	}
}
