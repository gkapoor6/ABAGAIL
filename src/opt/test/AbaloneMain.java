package opt.test;

public class AbaloneMain {

	public static void main(String[] args) {
			
		
//    	Vary popSize (150-300), toMate (0-100), toMutate (0-100) with 20 iterations
//    	population must be greater than toMate, toMutate
//    	Grid search
   	

	   	for (int i = 50; i < 150; i += 20) {
	   		for (int j = 10; j < 51; j += 20) {
	   			for (int k = 10; k < 51; k += 20) {
	   				// SA params won't be used 
	   				AbaloneGA ag = new AbaloneGA(20, 1E11, .95, i, j, k);
	   				ag.run();
	   			}
	   		}
	   	}

	   	
//    	Vary t, cooling with 500 iterations
//    	Grid search
	   	

   		for (double j = 1e1; j < 1e15; j *= 1e2) {
   			for (double k = 0.1; k < 1; k += 0.3) {
   				// GA params won't be used 
   				AbaloneSA ag = new AbaloneSA(20, j, k, 1,1,1);
   				ag.run();
   			}
   		}


	   	
		// Vary iterations with optimal hyperparams
		for (int k = 100; k < 5000; k += 500) {
			AbaloneSA as = new AbaloneSA(k, 1E11, .95, 70,30,50);
			as.run();
			AbaloneGA ag = new AbaloneGA(k, 1E11, .95, 80, 30, 20);
			ag.run();
			AbaloneRHC ar = new AbaloneRHC(k, 1E11, .95, 80, 30, 20);
			ar.run();
		}
   	
	}
}
