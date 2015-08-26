package DeepLearning;

import java.util.Random;

public class MLP {
    public int N;
    public int n_in;
    public int n_hidden;
    public int n_out;
    public HiddenLayer hiddenLayer;
    public LogisticRegression logisticLayer;
    public Random rng;


    public MLP(int N, int n_in, int n_hidden, int n_out, Random rng) {

        this.N = N;
        this.n_in = n_in;
        this.n_hidden = n_hidden;
        this.n_out = n_out;

        if (rng == null)rng = new Random(1234);
        this.rng = rng;

        // construct hiddenLayer
        this.hiddenLayer = new HiddenLayer(N, n_in, n_hidden, null, null, rng, "tanh");

        // construct logisticLayer
        this.logisticLayer = new LogisticRegression(N, n_hidden, n_out);
    }


    public void train(double[][] train_X, int[][] train_Y, double lr) {
        double[] hidden_layer_input;
        double[] logistic_layer_input;
        double[] dy;

        for(int n=0; n<N; n++) {
            hidden_layer_input = new double[n_in];
            logistic_layer_input = new double[n_hidden];

            for(int j=0; j<n_in; j++) hidden_layer_input[j] = train_X[n][j];

            // forward hiddenLayer
            hiddenLayer.forward(hidden_layer_input, logistic_layer_input);

            // forward and backward logisticLayer
            // dy = new double[n_out];  // define delta of y for backpropagation
            dy = logisticLayer.train(logistic_layer_input, train_Y[n], lr); //, dy);

            // backward hiddenLayer
            hiddenLayer.backward(hidden_layer_input, null, logistic_layer_input, dy, logisticLayer.W, lr);

        }
    }

    public void predict(double[] x, double[] y) {
        double[] logistic_layer_input = new double[n_hidden];
        hiddenLayer.forward(x, logistic_layer_input);
        logisticLayer.predict(logistic_layer_input, y);
    }



    private static void test_mlp() {
        Random rng = new Random(123);

        double learning_rate = 0.1;
        int n_epochs = 5000;

        int train_N = 4;
        int test_N = 4;
        int n_in = 2;
        int n_hidden = 3;
        int n_out = 2;

        double[][] train_X = {
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.},
        };

        int[][] train_Y = {
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1},
        };

        // construct MLP
        MLP classifier = new MLP(train_N, n_in, n_hidden, n_out, rng);

        // train
        for(int epoch=0; epoch<n_epochs; epoch++) {
            classifier.train(train_X, train_Y, learning_rate);
        }

        // test data
        double[][] test_X = {
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.},
        };

        double[][] test_Y = new double[test_N][n_out];


        // test
        for(int i=0; i<test_N; i++) {
            classifier.predict(test_X[i], test_Y[i]);
            for(int j=0; j<n_out; j++) {
                System.out.print(test_Y[i][j] + " ");
            }
            System.out.println();
        }

    }

    public static void main(String[] args) {
        test_mlp();
    }
}
