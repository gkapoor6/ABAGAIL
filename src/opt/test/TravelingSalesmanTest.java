package opt.test;

import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    public static int maxN = 200;
    public static int N = 80;
    public static int maxIterations = 2000;
    private static List<String> lines = new ArrayList<>();
    private static List<String> lines1 = new ArrayList<>();
    
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
    	
    	/**
        for(int l = 0; l < maxIterations; l += 300) {
            Random random = new Random();
            // create the random points
            double[][] points = new double[N][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextDouble();
                points[i][1] = random.nextDouble();   
            }
            
            
            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            
            // for mimic we use a sort encoding
            ef = new TravelingSalesmanSortEvaluationFunction(points);
            int[] ranges = new int[N];
            Arrays.fill(ranges, N);
            odd = new  DiscreteUniformDistribution(ranges);
            Distribution df = new DiscreteDependencyTree(.1, ranges); 
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            
        	System.out.println("Iterations = " + l);
        	
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, l);
            fit.train();
            
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .9, hcp);
            fit = new FixedIterationTrainer(sa, l);
            fit.train();
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(80, 20, 10, gap);
            fit = new FixedIterationTrainer(ga, l);
            fit.train();
     
            MIMIC mimic = new MIMIC(80, 60, pop);
            fit = new FixedIterationTrainer(mimic, l);
            fit.train();
            
            lines.add(l + ", " + ef.value(rhc.getOptimal()) + ", " + ef.value(sa.getOptimal()) +
            		", " + ef.value(ga.getOptimal()) + ", " + ef.value(mimic.getOptimal()));
        }
        
        try {
        	// won't work on any other machine - change path!!!
        	Path file = Paths.get("C:\\Users\\Geetika\\Documents\\tsp_iterations.csv");
            Files.write(file, lines, Charset.forName("UTF-8"));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        */
        
    	

        for(int m = 20; m < maxN; m += 50) {
      	
        	// Using default iterations for all
            Random random = new Random();
            // create the random points
            double[][] points = new double[m][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextDouble();
                points[i][1] = random.nextDouble();   
            }
            
            
            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(m);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            
            // for mimic we use a sort encoding
            ef = new TravelingSalesmanSortEvaluationFunction(points);
            int[] ranges = new int[m];
            Arrays.fill(ranges, m);
            odd = new  DiscreteUniformDistribution(ranges);
            Distribution df = new DiscreteDependencyTree(.1, ranges); 
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            
        	System.out.println("N = " + m);
        	
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
            fit.train();
            
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .9, hcp);
            fit = new FixedIterationTrainer(sa, 20000);
            fit.train();
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(10, 5, 25, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
     
            MIMIC mimic = new MIMIC(25, 15, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            
            lines1.add(m + ", " + ef.value(rhc.getOptimal()) + ", " + ef.value(sa.getOptimal()) +
            		", " + ef.value(ga.getOptimal()) + ", " + ef.value(mimic.getOptimal()));
         
        }
        try {
        	// won't work on any other machine - change path!!!
        	Path file = Paths.get("C:\\Users\\Geetika\\Documents\\tsp_problemsize.csv");
            Files.write(file, lines1, Charset.forName("UTF-8"));
            
        } catch (Exception e) {
            e.printStackTrace();
        }

        
    }
}
