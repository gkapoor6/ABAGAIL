package opt.test;

public class AbaloneMain {

	public static void main(String[] args) {
			
		
//    	Vary popSize (150-300), toMate (0-100), toMutate (0-100) with 20 iterations
//    	population must be greater than toMate, toMutate
//    	Grid search
   	
	   	for (int i = 50; i < 100; i += 10) {
	   		for (int j = 10; j < 51; j += 10) {
	   			for (int k = 10; k < 51; k += 10) {
	   				// SA params won't be used 
	   				AbaloneGA ag = new AbaloneGA(20, 1E11, .95, i, j, k);
	   				ag.run();
	   			}
	   		}
	   	}
	   	
//    	Vary t, cooling with 500 iterations
//    	Grid search
	   	
   		for (double j = 1e1; j < 1e14; j += 1e3) {
   			for (double k = 0.1; k < 1; k += 0.3) {
   				// GA params won't be used 
   				AbaloneSA ag = new AbaloneSA(20, j, k, 1,1,1);
   				ag.run();
   			}
   		}
	   	
		// Vary iterations with optimal params
//		AbaloneCV a1 = new AbaloneCV(100, 1E11, .95, 200, 100, 10);
//		a1.run();
   	
	}
}
