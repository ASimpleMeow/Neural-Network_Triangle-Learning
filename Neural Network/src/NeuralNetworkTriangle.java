import java.util.Arrays;
import java.util.Scanner;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.comp.layer.InputLayer;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

/**
 * A neural network that uses the Neuroph library.
 * Learns what a triangle is given a perimeter and a side.
 * 
 * @author Oleksandr Kononov
 * @version 18-02-2017
 *
 */
public class NeuralNetworkTriangle implements LearningEventListener{
	NeuralNetwork net;
	DataSet trainingSet;
	Scanner in;
	
	public static void main(String[] args){
		new NeuralNetworkTriangle().run();
	}
	
	public void run(){
		in = new Scanner(System.in);
		trainingSet = new DataSet(2,1);
		loadTrainingSet();
		
		// create multi layer perceptron
        
        net = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 2, 2, 1);
        
        // enable batch if using MomentumBackpropagation
        if( net.getLearningRule() instanceof MomentumBackpropagation )
        	((MomentumBackpropagation)net.getLearningRule()).setBatchMode(true);
        
        LearningRule learningRule = net.getLearningRule();
        learningRule.addListener(this);
        System.out.println("learning rule : "+learningRule);
        
        System.out.println("Training neural network ... ");
        net.learn(trainingSet);
        net.learn(trainingSet);
        net.learn(trainingSet);
        net.learn(trainingSet);
        
        //Test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(net, trainingSet);
        
        runMenu();
	}
	
	public void runMenu(){
		System.out.println("\n\n\n\n\n\n\nTRIANGLE NEURAL NETWORK");
		System.out.println("Type the sides and perimeter of a shape and I'll"
				+ " tell you if it's a triangle! Type '0' to exit");
		double sides = 1, perimeter = 1;
		do{
			System.out.print("Sides : ");
			sides = in.nextInt();
			System.out.print("Perimeter : ");
			perimeter = in.nextInt();
			net.setInput(new double[] {normalize(sides),normalize(perimeter)});
			net.calculate();
			double out[] = net.getOutput();
			if(out[0] > 0.8)
				System.out.println("This is a triangle");
			else
				System.out.println("This is not a triangle");
			
		}while(sides != 0 || perimeter != 0);
	}
	
	private double normalize(double x){
		return 1/(1+Math.exp(x));
	}
	
	@Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
    } 
	
	public static void testNeuralNetwork(NeuralNetwork net, DataSet testSet){
		for(DataSetRow testSetRow : testSet.getRows()) {
			net.setInput(testSetRow.getInput());
			net.calculate();
            double[] networkOutput = net.getOutput();

            System.out.print("Input: " + Arrays.toString( testSetRow.getInput() ) );
            System.out.println(" Output: " + Arrays.toString( networkOutput) );
        }
	}
	
	private void loadTrainingSet(){
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(05)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(15)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(10)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(4),normalize(25)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(5),normalize(20)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(40)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(10),normalize(17)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(7),normalize(05)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(6),normalize(20)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(4),normalize(25)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(05)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(30)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(25)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(05)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(15)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(10)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(4),normalize(15)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(5),normalize(25)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(40)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(22),normalize(30)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(4),normalize(15)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(4),normalize(10)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(4),normalize(05)},
				new double[] {0}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(05)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(30)},
				new double[] {1}));
		trainingSet.addRow(new DataSetRow(new double[] {normalize(3),normalize(25)},
				new double[] {1}));
	}
}
