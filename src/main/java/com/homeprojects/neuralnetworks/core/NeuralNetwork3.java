package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.homeprojects.neuralnetworks.core.Utils.sigmoid;

public class NeuralNetwork3 {

    private final Random random;

    private final SimpleMatrix inputs;

    private final SimpleMatrix outputs;

    private final List<Layer> layers;

    private final double learningRate;

    public NeuralNetwork3(SimpleMatrix inputs, SimpleMatrix outputs, int[] layersNeuronsCount, double learningRate) {
        this.outputs = outputs;
        this.learningRate = learningRate;
        if (outputs.getNumCols() > 1) {
            throw new IllegalArgumentException(String.format("outputs array should have only one row stating only one output, got %s", outputs.getNumRows()));
        }
        this.random = new Random(10);
        this.inputs = inputs;
        this.layers = new ArrayList<>();

        for (int neuronsCount : layersNeuronsCount) {
            SimpleMatrix w = Utils.random(inputs.getNumCols(), neuronsCount, random);
            SimpleMatrix b = Utils.random(neuronsCount, 1, random);
            SimpleMatrix a = SimpleMatrix.filled(w.getNumCols(), 1, 0);
            layers.add(new Layer(w, b, a));
            inputs = a.transpose();
        }
    }

    public void start() {
        iterate();
    }

    private void iterate() {
        for (int i = 0; i < inputs.getNumRows(); i++) {
            forward(inputs.getRow(i));
            backpropagate(i);
        }
    }

    private void forward(int index) {
        forward(inputs.getRow(index));
    }

    private void forward(SimpleMatrix row) {
        SimpleMatrix x = row;
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            SimpleMatrix w = layer.w();
            SimpleMatrix b = layer.b();

            LoggerUtils.printDims("x", x);
            LoggerUtils.printDims("w", w);

            SimpleMatrix temp = x.mult(w).transpose();
            LoggerUtils.printDims("x * w", temp);
            LoggerUtils.printDims("b", b);

            temp = temp.plus(b);
            temp = Utils.elementOp(temp, (r, c, v) -> sigmoid(v));
            LoggerUtils.printDims("a", temp);
            x = temp.transpose();
            layers.set(i, new Layer(layer.w(), layer.b(), temp));
            LoggerUtils.printLine();
        }
    }

    private void backpropagate(int index) {
        Map<Integer, SimpleMatrix> cache = new HashMap<>();
        handleLastLayer(index, cache);

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer layer = layers.get(i);
            SimpleMatrix temp = cache.get(i + 1);
            temp = temp.mult(layers.get(i + 1).w());
            SimpleMatrix a = Utils.elementOp(layer.a(), Utils::sigmoidDerivative);
            SimpleMatrix db = temp.mult(a);
            cache.put(i, db);

            SimpleMatrix dw = null;
            if (i - 1 == -1) {
                dw = db.mult(inputs.getRow(index));
            } else {
                dw = db.mult(layers.get(i - 1).a());
            }

            dw = Utils.elementOp(dw, v -> v * learningRate);
            db = Utils.elementOp(db, v -> v * learningRate);

            SimpleMatrix newW = layer.w().plus(dw);
            SimpleMatrix newB = layer.b().plus(db);
            layers.set(layers.size() - 1, new Layer(newW, newB, layer.a()));
        }
    }

    private void handleLastLayer(int index, Map<Integer, SimpleMatrix> cache) {
        Layer layer = layers.getLast();
        SimpleMatrix a = layer.a();
        SimpleMatrix db = Utils.elementOp(a.minus(outputs.getRow(index)), Utils::sigmoidDerivative);

        LoggerUtils.printDims("db", db);

        cache.put(layers.size() - 1, db);
        LoggerUtils.printDims("layers.get(layers.size() - 2).a()", layers.get(layers.size() - 2).a());
        SimpleMatrix dw = db.mult(layers.get(layers.size() - 2).a());
        dw = Utils.elementOp(dw, v -> v * learningRate);
        db = Utils.elementOp(db, v -> v * learningRate);

        SimpleMatrix newW = layer.w().plus(dw);
        SimpleMatrix newB = layer.b().plus(db);
        layers.set(layers.size() - 1, new Layer(newW, newB, layer.a()));
    }
}
