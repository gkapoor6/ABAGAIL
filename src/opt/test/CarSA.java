package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.net.URL;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying the quality of the car data set
 *
 * @author Hannah Lau, Geetika Kapoor
 * @version 1.1
 */
public class CarSA {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 7, hiddenLayer = 5, outputLayer = 1;
    private static int trainingIterations;
   
    private static double t,cooling;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"SA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    // Constructor
    public CarSA(int trainingIterations, double t, double cooling) {
    	CarSA.t = t;
    	CarSA.cooling = cooling;
    	CarSA.trainingIterations = trainingIterations;
    }
    
    public void run() {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new SimulatedAnnealing(CarSA.t, CarSA.cooling, nnop[0]);

        System.out.println( "Temperature: " + CarSA.t + ", Cooling: " + CarSA.cooling);
        for(int i = 0; i < oa.length; i++) {
        	
        	// TRAINING
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]);
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            double tp = 0.0, tn = 0.0, fp = 0.0, fn = 0.0;
            double accuracy = 0.0, precision = 0.0, recall = 0.0, f1 = 0.0;
            
            // TESTING
            start = System.nanoTime();
            double testingError = 0.0;
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());
                
                Instance actual1 = new Instance(networks[i].getOutputValues());
                actual1.setLabel(new Instance(actual));
                
                testingError += measure.value(instances[j].getLabel(), actual1); 
                // calculate F1 score - code by PHPCoderBlog 
                // at (https://phpcoderblog.wordpress.com/2017/11/02/how-to-calculate-accuracy-precision-recall-and-f1-score-deep-learning-precision-recall-f-score-calculating-precision-recall-python-precision-recall-scikit-precision-recall-ml-metrics-to-use-bi/)
                
                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;   
                // if predicted == 1
                if (Math.abs(predicted - 1) < 0.5) 
                {
                    if (Math.abs(predicted - actual) < 0.5)
                    {
                    	tp++;
                    } else {
                    	fp++;
                    }
                }
                // if predicted == 0
                else {
                    if (Math.abs(predicted - actual) < 0.5)
                    {
                    	tn++;
                    } else {
                    	fn++;
                    }
                }
            }
            
            testingError = testingError/instances.length;
            
            
            // a ratio of correctly predicted observation to the total observations
            accuracy = (tp + tn)/(tp + tn + fp + fn);
         
            // precision is "how useful the search results are"
            precision = tp / (tp + fp);
            
            // recall is "how complete the results are"
            recall = tp / (tp + fn);
         
            f1 = 2 / ((1 / precision) + (1 / recall));
            
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "\nF1 measure: " + df.format(f1) +
                        "\nTesting Error: " + df.format(testingError) + ".\n";
        } 

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
    	double trainingError = 0.0;
        	
    	// EVERY ITERATION
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();           
            double error = 0;
            
            // EVERY INSTANCE
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                // adds error for each instance in the training iteration
                error += measure.value(output, example); 
            }
            
            // adds error for each training iteration
            trainingError += error/instances.length;
        }
        
        System.out.println("\nTraining error: " + df.format(trainingError/trainingIterations));
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[1727][][];

        try {
        	
        	URL path = CarTest.class.getResource("car_processed.csv");
            BufferedReader br = new BufferedReader(new FileReader(new File(path.getFile())));
            
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[6]; // 6 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 6; j++)
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
            // classifications range from 1 to 4; split into 4 labels
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
