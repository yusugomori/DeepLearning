package DeepLearning;

import java.util.Random;
import static DeepLearning.utils.*;

public class dA {
    public int n_visible;
    public int N;
    public int n_hidden;
    public double[][] W;
    public double[] hbias;
    public double[] vbias;
    public Random rng;


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
                    this.W[i][j] = uniform(-a, a, rng);
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
                tilde_x[i] = binomial(1, p, rng);
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

    private static void test_dA() {
        Random rng = new Random(123);

        double learning_rate = 0.1;
        double corruption_level = 0.3;
        int training_epochs = 100;

        int train_N = 10;
        int test_N = 2;
        int n_visible = 20;
        int n_hidden = 5;

        int[][] train_X = {
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}
        };

        dA da = new dA(train_N, n_visible, n_hidden, null, null, null, rng);

        // train
        for(int epoch=0; epoch<training_epochs; epoch++) {
            for(int i=0; i<train_N; i++) {
                da.train(train_X[i], learning_rate, corruption_level);
            }
        }

        // test data
        int[][] test_X = {
                {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
        };

        double[][] reconstructed_X = new double[test_N][n_visible];

        // test
        for(int i=0; i<test_N; i++) {
            da.reconstruct(test_X[i], reconstructed_X[i]);
            for(int j=0; j<n_visible; j++) {
                System.out.printf("%.5f ", reconstructed_X[i][j]);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        test_dA();
    }
}
