package com.homeprojects.neuralnetworks.manual;

import com.homeprojects.neuralnetworks.core.LoggerUtils;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

// Works like a charm
public class Single1HiddenNoSig {

    private static final Random RANDOM = new Random(10);

    public static void main(String[] args) {
        double[] inputs = new double[]  {1, -1, 2, -3, 5, -10, 6, 0};
        double[] outputs = new double[] {2, -2, 4, -6,10, -20, 12, 0};
//        double[] inputs = new double[]  {1, 0};
//        double[] outputs = new double[] {0, 1};
        Single1HiddenNoSig network = new Single1HiddenNoSig(inputs, outputs);

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

    public Single1HiddenNoSig(double[] inputs, double[] outputs) {
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
        for (int i = 0; i < 500; i++) {
            iterate();
            System.out.printf("cost = %f\n", cost());
        }
        System.out.printf("w1 = %f\nb1 = %f\nw2 = %f\nb2 = %f\n", w1, b1, w2, b2);
        LoggerUtils.printLine();

        for (int i = -5; i < 5; i++) {
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
        double db2 = 2 * (a2 - outputs[index]);
        double dw2 = db2 * a1;

        double db1 = db2 * w2;
        double dw1 = db1 * inputs[index];

        w1 = w1 + dw1 * learningRate;
        w2 = w2 + dw2 * learningRate;
        b1 = b1 + db1 * learningRate;
        b2 = b2 + db2 * learningRate;
    }

    private void forwardValue(double value) {
        a1 = w1 * value + b1;
        a2 = w2 * a1 + b2;
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
