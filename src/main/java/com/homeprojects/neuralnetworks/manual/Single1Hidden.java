package com.homeprojects.neuralnetworks.manual;

import com.homeprojects.neuralnetworks.core.Utils;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import static com.homeprojects.neuralnetworks.core.Utils.sigmoidDerivative;

// Works like a charm
public class Single1Hidden {

    private static final Random RANDOM = new Random(10);

    public static void main(String[] args) {
//        double[] inputs = new double[]  {1, -1, 2, -3, 5, -10, 1000, 0};
//        double[] outputs = new double[] {1,  0, 1,  0, 1,   0,    1, 1};
        double[] inputs = new double[]  {1, 0};
        double[] outputs = new double[] {0, 1};
        Single1Hidden network = new Single1Hidden(inputs, outputs);

        Instant start = Instant.now();
        network.start();
        System.out.printf("Time taken: %dms", Duration.between(start, Instant.now()).toMillis());
    }

    private final double[] inputs;

    private double w1;
    private double b1;
    private double a1;

    private double w2;
    private double b2;
    private double a2;

    private final double[] outputs;

    private final double learningRate = -0.001;

    public Single1Hidden(double[] inputs, double[] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
        w1 = random();
        b1 = random();
        w2 = random();
        b2 = random();
    }

    private static double random() {
        return RANDOM.nextDouble();
    }

    public void start() {
        for (int i = 0; i < 50000000; i++) {
            iterate();
        }
        System.out.printf("cost = %f\n", cost());


        for (int i = 0; i < 2; i++) {
            forwardValue(i);
            System.out.printf("%d: %f\n", i, a2);
        }
    }

    private void iterate() {
        for (int i = 0; i < inputs.length; i++) {
            forward(i);
            backpropagate(i);
        }
    }

    private void backpropagate(int index) {
        double db2 = 2 * (a2 - outputs[index]) * sigmoidDerivative(a2);
        double dw2 = db2 * a1;

        double db1 = db2 * w2 * sigmoidDerivative(a1);
        double dw1 = db1 * inputs[index];

        w1 = w1 + dw1 * learningRate;
        w2 = w2 + dw2 * learningRate;
        b1 = b1 + db1 * learningRate;
        b2 = b2 + db2 * learningRate;
    }

    private void forwardValue(double value) {
        double a1Temp = w1 * value + b1;
        a1 = Utils.sigmoid(a1Temp);

        double a2Temp = w2 * a1 + b2;
        a2 = Utils.sigmoid(a2Temp);
    }

    private void forward(int index) {
        forwardValue(inputs[index]);
    }

    private double cost() {
        double cost = 0;
        for (int i = 0; i < inputs.length; i++) {
            forward(i);
            cost = cost + Math.pow(a2 - outputs[i], 2);
        }
        return cost;
    }
}
