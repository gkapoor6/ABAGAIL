package opt.test;

import java.util.Arrays;
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
public class TSPGA {
    /** The n value */
    private static final int N = 50;
    private static final int iterations = 100;
    public static int trainingIterations = 200;
    private static int popSize;static int toMate; static int toMutate;

    public TSPGA(int popSize, int toMate, int toMutate) {
    	TSPGA.popSize = popSize;
    	TSPGA.toMate = toMate;
    	TSPGA.toMutate = toMutate;
    }
    
    public static void run() {
 
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        
        // for SA, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        for(int i = 0; i < iterations; i++)
        {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(TSPGA.popSize, TSPGA.toMate, TSPGA.toMutate, gap);
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, trainingIterations);
            fit.train();
            System.out.println(ef.value(ga.getOptimal()));
        }
    }
}

