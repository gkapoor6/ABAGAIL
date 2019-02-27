package opt.test;

public class TSPHyperMain {

	public static void main(String[] args) {
		
		for (int i = 100; i < 1001; i += 300) {
			
			TSPRHC tsprhc = new TSPRHC(i);
			tsprhc.run();
			
			for (int j = 50; j < 401; j += 100) {
				for (double k = 0.1; k < 1; k += .2) {
					TSPSA tspsa = new TSPSA(i ,j, k);
					tspsa.run();
				}
			}
	    		    	
		}
		
    	for (int i = 100; i < 501; i += 200) {
    		for (int j = 0; j < 101; j += 25) {
    			for (int k = 0; k < 101; k += 25) {
    				TSPGA tspga = new TSPGA(i, j, k);
    				tspga.run();
    			}
    		}
    	}
    	
		for (int j = 50; j < 401; j += 100) {
			for (int m = 0; m < 201; m += 25) {
				TSPMC tspsa = new TSPMC(j, m);
				tspsa.run();
			}
		}
    	
		

	}
}
