import java.util.Random;

public class dA {
	public int N;
	public int n_visible;
	public int n_hidden;
	public double[][] W;
	public double[] hbias;
	public double[] vbias;
	public Random rng;
	
	
	public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}
	
	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;
		
		int c = 0;
		double r;
		
		for(int i=0; i<n; i++) {
			r = rng.nextDouble();
			if (r < p) c++;
		}
		
		return c;
	}
	
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}

	public dA(int N, int n_visible, int n_hidden, 
			double[][] W, double[] hbias, double[] vbias, Random rng) {
		this.N = N;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;

		if(rng == null)	this.rng = new Random(1234);
		else this.rng = rng;
				
		if(W == null) {
			this.W = new double[this.n_hidden][this.n_visible];
			double a = 1.0 / this.n_visible;
			
			for(int i=0; i<this.n_hidden; i++) {
				for(int j=0; j<this.n_visible; j++) {
					this.W[i][j] = uniform(-a, a); 
				}
			}	
		} else {
			this.W = W;
		}
		
		if(hbias == null) {
			this.hbias = new double[this.n_hidden];
			for(int i=0; i<this.n_hidden; i++) this.hbias[i] = 0;
		} else {
			this.hbias = hbias;
		}
		
		if(vbias == null) {
			this.vbias = new double[this.n_visible];
			for(int i=0; i<this.n_visible; i++) this.vbias[i] = 0;
		} else {
			this.vbias = vbias;
		}	
	}
	
	public void get_corrupted_input(int[] x, int[] tilde_x, double p) {
		for(int i=0; i<n_visible; i++) {
			if(x[i] == 0) {
				tilde_x[i] = 0;
			} else {
				tilde_x[i] = binomial(1, p);
			}
		}
	}
	
	// Encode
	public void get_hidden_values(int[] x, double[] y) {
		for(int i=0; i<n_hidden; i++) {
			y[i] = 0;
			for(int j=0; j<n_visible; j++) {
				y[i] += W[i][j] * x[j];
			}
			y[i] += hbias[i];
			y[i] = sigmoid(y[i]);
		}
	}
	
	// Decode
	public void get_reconstructed_input(double[] y, double[] z) {
		for(int i=0; i<n_visible; i++) {
			z[i] = 0;
			for(int j=0; j<n_hidden; j++) {
				z[i] += W[j][i] * y[j];
			}
			z[i] += vbias[i];
			z[i] = sigmoid(z[i]);
		}
	}
	
	public void train(int[] x, double lr, double corruption_level) {
		int[] tilde_x = new int[n_visible];
		double[] y = new double[n_hidden];
		double[] z = new double[n_visible];
		
		double[] L_vbias = new double[n_visible];
		double[] L_hbias = new double[n_hidden];
		
		double p = 1 - corruption_level;
		
		get_corrupted_input(x, tilde_x, p);
		get_hidden_values(tilde_x, y);
		get_reconstructed_input(y, z);
		
		// vbias
		for(int i=0; i<n_visible; i++) {
			L_vbias[i] = x[i] - z[i];
			vbias[i] += lr * L_vbias[i] / N;
		}
		
		// hbias
		for(int i=0; i<n_hidden; i++) {
			L_hbias[i] = 0;
			for(int j=0; j<n_visible; j++) {
				L_hbias[i] += W[i][j] * L_vbias[j];
			}
			L_hbias[i] *= y[i] * (1 - y[i]);
			hbias[i] += lr * L_hbias[i] / N;
		}
		
		// W
		for(int i=0; i<n_hidden; i++) {
			for(int j=0; j<n_visible; j++) {
				W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / N;
			}
		}
	}
	
	public void reconstruct(int[] x, double[] z) {
		double[] y = new double[n_hidden];
		
		get_hidden_values(x, y);
		get_reconstructed_input(y, z);
	}	
}
