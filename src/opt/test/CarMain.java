package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import opt.test.*;
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
public class CarMain {
	
    /**
     * @param args
     */
    public static void main(String[] args) {

    	// Vary #Iterations
    	/**
   	CarTestCrossValidation car1 = new CarTestCrossValidation(1, 1E11, .95, 200, 100, 10);
   	car1.run();
   	CarTestCrossValidation car2 = new CarTestCrossValidation(20, 1E11, .95, 200, 100, 10);
   	car2.run();
   	CarTestCrossValidation car3 = new CarTestCrossValidation(50, 1E11, .95, 200, 100, 10);
   	car3.run();
   	CarTestCrossValidation car4 = new CarTestCrossValidation(100, 1E11, .95, 200, 100, 10);
   	car4.run();
   	CarTestCrossValidation car5 = new CarTestCrossValidation(500, 1E11, .95, 200, 100, 10);
   	car5.run();
   	CarTestCrossValidation car6 = new CarTestCrossValidation(1000, 1E11, .95, 200, 100, 10);
   	car6.run();
   	*/
    	
    	// Vary Temperature (1E9 - 1E15) AND Cooling Rate for SA (0.2 - 0.95) with 1000 iterations
    	// Run a Grid Search
    	/**	
	CarSA car1 = new CarSA(1000, 1E9, .2);
	car1.run();
   	CarSA car2 = new CarSA(1000, 1E10, .2);
   	car2.run();
   	CarSA car3 = new CarSA(1000, 1E11, .2);
   	car3.run();
   	CarSA car4 = new CarSA(1000, 1E12, .2);
   	car4.run();
   	CarSA car5 = new CarSA(1000, 1E13, .2);
   	car5.run();
   	CarSA car6 = new CarSA(1000, 1E15, .2);
   	car6.run();
	
   	CarSA car12 = new CarSA(1000, 1E9, .4);
   	car12.run();
   	CarSA car22 = new CarSA(1000, 1E10, .4);
   	car22.run();
   	CarSA car32 = new CarSA(1000, 1E11, .4);
   	car32.run();
   	CarSA car42 = new CarSA(1000, 1E12, .4);
   	car42.run();
   	CarSA car52 = new CarSA(1000, 1E13, .4);
   	car52.run();
   	CarSA car62 = new CarSA(1000, 1E15, .4);
   	car62.run();
   	
   	CarSA car13 = new CarSA(1000, 1E9, .6);
   	car13.run();
   	CarSA car23 = new CarSA(1000, 1E10, .6);
   	car23.run();
   	CarSA car33 = new CarSA(1000, 1E11, .6);
   	car33.run();
   	CarSA car43 = new CarSA(1000, 1E12, .6);
   	car43.run();
   	CarSA car53 = new CarSA(1000, 1E13, .6);
   	car53.run();
   	CarSA car63 = new CarSA(1000, 1E15, .6);
   	car63.run();
   	
   	CarSA car14 = new CarSA(1000, 1E9, .8);
   	car14.run();
   	CarSA car24 = new CarSA(1000, 1E10, .8);
   	car24.run();
   	CarSA car34 = new CarSA(1000, 1E11, .8);
   	car34.run();
   	CarSA car44 = new CarSA(1000, 1E12, .8);
   	car44.run();
   	CarSA car54 = new CarSA(1000, 1E13, .8);
   	car54.run();
   	CarSA car64 = new CarSA(1000, 1E15, .8);
   	car64.run();
   	
   	CarSA car15 = new CarSA(1000, 1E9, .99);
   	car15.run();
   	CarSA car25 = new CarSA(1000, 1E10, .99);
   	car25.run();
   	CarSA car35 = new CarSA(1000, 1E11, .99);
   	car35.run();
   	CarSA car45 = new CarSA(1000, 1E12, .99);
   	car45.run();
   	CarSA car55 = new CarSA(1000, 1E13, .99);
   	car55.run();
   	CarSA car65 = new CarSA(1000, 1E15, .99);
   	car65.run();
   	*/    	
 	
//    	Vary popSize (150-300), toMate (0-100), toMutate (0-100) with 20 iterations
//    	population must be greater than toMate, toMutate
//    	Grid search
   	/**
   	for (int i = 50; i < 100; i += 10) {
   		for (int j = 10; j < 51; j += 10) {
   			for (int k = 10; k < 51; k += 10) {
   				CarGA car = new CarGA(100, i, j, k);
   				car.run();
   			}
   		}
   	}
   	*/
//   	CarGA car1 = new CarGA(100, 100, 25, 20);	
//   	car1.run();
    	
    }   
}
