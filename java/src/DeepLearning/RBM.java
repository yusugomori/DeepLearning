package DeepLearning;

import java.util.Random;
import static DeepLearning.utils.*;

public class RBM {
    public int N;
    public int n_visible;
    public int n_hidden;
    public double[][] W;
    public double[] hbias;
    public double[] vbias;
    public Random rng;


    public RBM(int N, int n_visible, int n_hidden,
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


    public void contrastive_divergence(int[] input, double lr, int k) {
        double[] ph_mean = new double[n_hidden];
        int[] ph_sample = new int[n_hidden];
        double[] nv_means = new double[n_visible];
        int[] nv_samples = new int[n_visible];
        double[] nh_means = new double[n_hidden];
        int[] nh_samples = new int[n_hidden];
		
		/* CD-k */
        sample_h_given_v(input, ph_mean, ph_sample);

        for(int step=0; step<k; step++) {
            if(step == 0) {
                gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
            } else {
                gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
            }
        }

        for(int i=0; i<n_hidden; i++) {
            for(int j=0; j<n_visible; j++) {
                // W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
                W[i][j] += lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
            }
            hbias[i] += lr * (ph_sample[i] - nh_means[i]) / N;
        }


        for(int i=0; i<n_visible; i++) {
            vbias[i] += lr * (input[i] - nv_samples[i]) / N;
        }

    }


    public void sample_h_given_v(int[] v0_sample, double[] mean, int[] sample) {
        for(int i=0; i<n_hidden; i++) {
            mean[i] = propup(v0_sample, W[i], hbias[i]);
            sample[i] = binomial(1, mean[i], rng);
        }
    }

    public void sample_v_given_h(int[] h0_sample, double[] mean, int[] sample) {
        for(int i=0; i<n_visible; i++) {
            mean[i] = propdown(h0_sample, i, vbias[i]);
            sample[i] = binomial(1, mean[i], rng);
        }
    }

    public double propup(int[] v, double[] w, double b) {
        double pre_sigmoid_activation = 0.0;
        for(int j=0; j<n_visible; j++) {
            pre_sigmoid_activation += w[j] * v[j];
        }
        pre_sigmoid_activation += b;
        return sigmoid(pre_sigmoid_activation);
    }

    public double propdown(int[] h, int i, double b) {
        double pre_sigmoid_activation = 0.0;
        for(int j=0; j<n_hidden; j++) {
            pre_sigmoid_activation += W[j][i] * h[j];
        }
        pre_sigmoid_activation += b;
        return sigmoid(pre_sigmoid_activation);
    }

    public void gibbs_hvh(int[] h0_sample, double[] nv_means, int[] nv_samples, double[] nh_means, int[] nh_samples) {
        sample_v_given_h(h0_sample, nv_means, nv_samples);
        sample_h_given_v(nv_samples, nh_means, nh_samples);
    }


    public void reconstruct(int[] v, double[] reconstructed_v) {
        double[] h = new double[n_hidden];
        double pre_sigmoid_activation;

        for(int i=0; i<n_hidden; i++) {
            h[i] = propup(v, W[i], hbias[i]);
        }

        for(int i=0; i<n_visible; i++) {
            pre_sigmoid_activation = 0.0;
            for(int j=0; j<n_hidden; j++) {
                pre_sigmoid_activation += W[j][i] * h[j];
            }
            pre_sigmoid_activation += vbias[i];

            reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
        }
    }



    private static void test_rbm() {
        Random rng = new Random(123);

        double learning_rate = 0.1;
        int training_epochs = 1000;
        int k = 1;

        int train_N = 6;
        int test_N = 2;
        int n_visible = 6;
        int n_hidden = 3;

        // training data
        int[][] train_X = {
                {1, 1, 1, 0, 0, 0},
                {1, 0, 1, 0, 0, 0},
                {1, 1, 1, 0, 0, 0},
                {0, 0, 1, 1, 1, 0},
                {0, 0, 1, 0, 1, 0},
                {0, 0, 1, 1, 1, 0}
        };



        RBM rbm = new RBM(train_N, n_visible, n_hidden, null, null, null, rng);

        // train
        for(int epoch=0; epoch<training_epochs; epoch++) {
            for(int i=0; i<train_N; i++) {
                rbm.contrastive_divergence(train_X[i], learning_rate, k);
            }
        }

        // test data
        int[][] test_X = {
                {1, 1, 0, 0, 0, 0},
                {0, 0, 0, 1, 1, 0}
        };

        double[][] reconstructed_X = new double[test_N][n_visible];

        for(int i=0; i<test_N; i++) {
            rbm.reconstruct(test_X[i], reconstructed_X[i]);
            for(int j=0; j<n_visible; j++) {
                System.out.printf("%.5f ", reconstructed_X[i][j]);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        test_rbm();
    }

}
