package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import util.linalg.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneRHC {
    private static Instance[] instances = initializeInstances();
    private static List<String> lines = new ArrayList<>();
    private static int inputLayer = 7, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
	  
	// Arguments for algorithms
  private static double t,cooling;
  private static int popSize, toMate, toMutate;
	  
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"RHC"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");
    
	  public AbaloneRHC(int trainingIterations, double t, double cooling, int popSize, int toMate, int toMutate) {
	    	
		  AbaloneRHC.trainingIterations = trainingIterations;
		  AbaloneRHC.t = t;
		  AbaloneRHC.cooling = cooling;
		  AbaloneRHC.popSize = popSize;
		  AbaloneRHC.toMate = toMate;
		  AbaloneRHC.toMutate = toMutate;
	  }

    public void run() {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);
            
            lines.add(trainingIterations + ", " + 
            			df.format(correct/(correct+incorrect)*100) + ", " + df.format(trainingTime) + ", " + df.format(testingTime)); 
            
            System.out.print(oaNames[0] + "iterations = "+ trainingIterations + "\n");
        }
        
        try {
        	// won't work on any other machine - change path!!!
        	Path file = Paths.get("C:\\Users\\Geetika\\Documents\\arhc.csv");
            Files.write(file, lines, Charset.forName("UTF-8"));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[4177][][];

        try {
        	URL path = AbaloneTest.class.getResource("abalone.txt");
            BufferedReader br = new BufferedReader(new FileReader(new File(path.getFile())));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[7]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 7; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
        }

        return instances;
    }
}


