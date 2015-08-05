package DeepLearning;

import java.util.Random;

public class utils {
    public static double uniform(double min, double max, Random rng) {
        return rng.nextDouble() * (max - min) + min;
    }

    public static int binomial(int n, double p, Random rng) {
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

}
