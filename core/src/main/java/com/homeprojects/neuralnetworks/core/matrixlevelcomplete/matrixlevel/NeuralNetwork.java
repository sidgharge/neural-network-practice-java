package com.homeprojects.neuralnetworks.core.matrixlevelcomplete.matrixlevel;

import com.homeprojects.neuralnetworks.core.Layer;
import com.homeprojects.neuralnetworks.core.LoggerUtils;
import com.homeprojects.neuralnetworks.core.Utils;
import org.ejml.simple.SimpleMatrix;

import java.io.Serializable;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.homeprojects.neuralnetworks.core.Utils.sigmoid;

public class NeuralNetwork implements Serializable {

    private final Random random;

    private final SimpleMatrix inputs;

    private final SimpleMatrix outputs;

    private final List<Layer> layers;

    private final double learningRate;

    private final int batchSize;

    public NeuralNetwork(SimpleMatrix inputs, SimpleMatrix outputs, int[] layersNeuronsCount, double learningRate, int batchSize) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.random = new Random(10);
        this.layers = new ArrayList<>();

        for (int neuronsCount : layersNeuronsCount) {
            SimpleMatrix w = Utils.random(neuronsCount, inputs.getNumRows(), random);
            SimpleMatrix b = Utils.random(neuronsCount, 1, random);
//            SimpleMatrix w = SimpleMatrix.filled(neuronsCount, inputs.getNumRows(), 0.5);
//            SimpleMatrix b = SimpleMatrix.filled(neuronsCount, 1, 0.5);
            SimpleMatrix a = SimpleMatrix.filled(neuronsCount, inputs.getNumCols(), 0);

//            LoggerUtils.print(String.format("w%d", layers.size()), w);
//            LoggerUtils.print(String.format("x%d", layers.size() - 1), inputs);
//            LoggerUtils.print(String.format("b%d", layers.size()), b);
//            LoggerUtils.print(String.format("a%d", layers.size()), a);
//
//            LoggerUtils.printLine();

            layers.add(new Layer(w, b, a));
            inputs = a;
        }
    }

    public void start() {
        start(500, 10, 0.001);
    }

    public void start(int maxIterations, int stepsToPrint, double maxAccuracy) {
        Instant start = Instant.now();
        for (int i = 1; i <= maxIterations; i++) {
            iterate();
            if (i % stepsToPrint == 0) {
                double cost = cost(this.inputs, this.outputs);
                System.out.printf("cost(%d) = %.10f. Time taken: %ds\n", i, cost, Duration.between(start, Instant.now()).toSeconds());
                start = Instant.now();
                if (cost <= maxAccuracy) {
                    System.out.println("Achieved max accuracy, hence stopping the training");
                    break;
                }
            }
        }
        LoggerUtils.printLine();
    }

    public SimpleMatrix test(SimpleMatrix testInputs, SimpleMatrix testOutputs) {
        double cost = cost(testInputs, testOutputs);
//        SimpleMatrix matrix = testInputs.concatRows(layers.getLast().a());
        System.out.printf("cost(test) = %.10f", cost);
        return layers.getLast().a();
    }

    public SimpleMatrix test(SimpleMatrix testInputs) {
        forward(testInputs);
//        SimpleMatrix matrix = testInputs.concatRows(layers.getLast().a());
        return layers.getLast().a();
    }

    private void iterate() {
        int totalSamples = inputs.getNumCols();
        int iterations = totalSamples / batchSize;
        if (totalSamples % batchSize != 0) {
            iterations++;
        }
        for (int i = 0; i < iterations; i++) {
            int c0 = i * batchSize;
            int c1 = (i + 1) * batchSize;
            if (c1 > inputs.getNumCols()) {
                c1 = inputs.getNumCols();
            }
            SimpleMatrix batchInputs = inputs.extractMatrix(0, inputs.getNumRows(), c0, c1);
            SimpleMatrix batchOutputs = outputs.extractMatrix(0, outputs.getNumRows(), c0, c1);
            forward(batchInputs);
            backpropagate(batchInputs, batchOutputs);
        }
    }

    public double cost() {
        return cost(inputs, outputs);
    }

    private double cost(SimpleMatrix inputs, SimpleMatrix outputs) {
        double cost = 0;
        forward(inputs);
        SimpleMatrix yHat = layers.getLast().a();
        for (int c = 0; c < outputs.getNumCols(); c++) {
            for (int r = 0; r < outputs.getNumRows(); r++) {
                double diff = yHat.get(r, c) - outputs.get(r, c);
                cost += Math.pow(diff, 2);
            }
        }
        return cost;
    }

    private void forward(SimpleMatrix batchInputs) {
        SimpleMatrix x = batchInputs;
        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            SimpleMatrix w = layer.w();
            SimpleMatrix b = layer.b();

            SimpleMatrix temp = w.mult(x);

            for (int r = 0; r < temp.getNumRows(); r++) {
                for (int c = 0; c < temp.getNumCols(); c++) {
                    double tb = temp.get(r, c) + b.get(r, 0);
                    temp.set(r, c, tb);
                }
            }

            temp = Utils.elementOp(temp, (r, c, v) -> sigmoid(v));
            layers.set(l, new Layer(layer.w(), layer.b(), temp));
            x = temp;
        }
    }

    private void backpropagate(SimpleMatrix batchInputs, SimpleMatrix batchOutputs) {
        Map<String, SimpleMatrix> cache = new HashMap<>();
        lastLayerDerivatives(cache, batchInputs, batchOutputs);

        for (int l = layers.size() - 2; l >= 0; l--) {
            currentLayerDerivatives(l, cache, batchInputs);
        }
        for (int l = layers.size() - 1; l >= 0; l--) {
            currentLayerAdjustments(l, cache);
        }
    }

    private void currentLayerAdjustments(int l, Map<String, SimpleMatrix> cache) {
        SimpleMatrix w = layers.get(l).w();
        SimpleMatrix b = layers.get(l).b();

        SimpleMatrix dw = cache.get(derivKey(l));

//        LoggerUtils.printDims("dw" + l, dw);

        for (int r = 0; r < w.getNumRows(); r++) {
            for (int c = 0; c < w.getNumCols(); c++) {
                double oldVal = w.get(r, c);
                double dVal = dw.get(r, c);
                double newVal = oldVal - (dVal * learningRate);
//                System.out.printf("old val: %.10f, new val: %.10f\n", oldVal, newVal);
                w.set(r, c, newVal);
            }
        }

        SimpleMatrix db = cache.get(deltaKey(l));
        for (int r = 0; r < db.getNumRows(); r++) {
            double dVal = 0;
            for (int c = 0; c < db.getNumCols(); c++) {
                dVal += db.get(r, c);
            }
//            dVal /= db.getNumCols();
            double oldVal = b.get(r, 0);
            double newVal = oldVal - (dVal * learningRate);
            b.set(r, 0, newVal);
        }
    }


    // n: sample number/row
    // l: current layer
    // j: jth neuron in current layer
    // k: kth neuron in next layer
    private void currentLayerDerivatives(int l, Map<String, SimpleMatrix> cache, SimpleMatrix batchInputs) {
        SimpleMatrix dzda = layers.get(l + 1).w();
        SimpleMatrix deltaMatrix = cache.get(deltaKey(l + 1));
        SimpleMatrix a = layers.get(l).a();
        SimpleMatrix dzdw = (l == 0) ? batchInputs : layers.get(l - 1).a();

        SimpleMatrix temp = dzda.transpose().mult(deltaMatrix);

        for (int r = 0; r < temp.getNumRows(); r++) {
            for (int c = 0; c < temp.getNumCols(); c++) {
                double val = temp.get(r, c) * a.get(r, c) * (1 - a.get(r, c));
                temp.set(r, c, val);
            }
        }
        cache.put(deltaKey(l), temp);

        temp = temp.mult(dzdw.transpose());
        cache.put(derivKey(l), temp);
    }

    private void lastLayerDerivatives(Map<String, SimpleMatrix> cache, SimpleMatrix batchInputs, SimpleMatrix batchOutputs) {
        SimpleMatrix y = batchOutputs;
        Layer layer = layers.getLast();
        SimpleMatrix a = layer.a();
        int l = layers.size() - 1;

        SimpleMatrix deltaMatrix = new SimpleMatrix(a.getNumRows(), a.getNumCols());
        for (int r = 0; r < a.getNumRows(); r++) {
            for (int c = 0; c < a.getNumCols(); c++) {
                double aVal = a.get(r, c);
                double yVal = y.get(r, c);
                double delta = 2 * (aVal - yVal) * aVal * (1 - aVal);
                deltaMatrix.set(r, c, delta);
            }
        }
        cache.put(deltaKey(l), deltaMatrix);
//        LoggerUtils.printDims("delta", deltaMatrix);

        SimpleMatrix prevA = l == 0 ? batchInputs : layers.get(l - 1).a();
//        LoggerUtils.printDims(String.format("a%d", l - 1), prevA);

        SimpleMatrix dw = deltaMatrix.mult(prevA.transpose());
        cache.put(derivKey(l), dw);
    }

    private String derivKey(int l) {
        return String.format("deriv_%d", l);
    }

    private String deltaKey(int l) {
        return String.format("delta_%d", l);
    }
}
