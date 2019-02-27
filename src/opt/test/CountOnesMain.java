package opt.test;

import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesMain {
    /** The n value */
    public static int maxN = 1000;
    public static int N = 80;
    public static int maxIterations = 200;
    private static List<String> lines = new ArrayList<>();
    private static List<String> lines1 = new ArrayList<>();
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
       
        for(int l = 0; l < maxIterations; l += 300) {
        	
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new CountOnesEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges); 
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            
        	System.out.println("Iterations = " + l);
        	
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, l);
            fit.train();
            System.out.println(ef.value(rhc.getOptimal()));
            
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, l);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()));
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, l);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()));
            
            MIMIC mimic = new MIMIC(50, 10, pop);
            fit = new FixedIterationTrainer(mimic, l);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()));
            
            lines.add(l + ", " + ef.value(rhc.getOptimal()) + ", " + ef.value(sa.getOptimal()) +
            		", " + ef.value(ga.getOptimal()) + ", " + ef.value(mimic.getOptimal()));
        }
        
        try {
        	// won't work on any other machine - change path!!!
        	Path file = Paths.get("C:\\Users\\Geetika\\Documents\\co1.csv");
            Files.write(file, lines, Charset.forName("UTF-8"));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        
        for(int m = 60; m < maxN; m += 30) {
      	
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new CountOnesEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges); 
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            
        	System.out.println("N = " + m);
        	
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200);
            fit.train();
            System.out.println(ef.value(rhc.getOptimal()));
            
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200);
            fit.train();
            System.out.println(ef.value(sa.getOptimal()));
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, 300);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()));
            
            MIMIC mimic = new MIMIC(50, 10, pop);
            fit = new FixedIterationTrainer(mimic, 100);
            fit.train();
            System.out.println(ef.value(mimic.getOptimal()));
            
            lines.add(m + ", " + ef.value(rhc.getOptimal()) + ", " + ef.value(sa.getOptimal()) +
            		", " + ef.value(ga.getOptimal()) + ", " + ef.value(mimic.getOptimal()));
         
        }
        try {
        	// won't work on any other machine - change path!!!
        	Path file = Paths.get("C:\\Users\\Geetika\\Documents\\co2.csv");
            Files.write(file, lines1, Charset.forName("UTF-8"));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}