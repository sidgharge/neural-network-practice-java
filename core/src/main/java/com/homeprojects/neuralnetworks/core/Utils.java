package com.homeprojects.neuralnetworks.core;

import com.homeprojects.neuralnetworks.core.matrixlevel.NeuralNetworkMatrix;
import org.ejml.simple.SimpleMatrix;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.DoubleFunction;
import java.util.function.ToDoubleBiFunction;

public class Utils {

    public static interface ElementOp {
        double op(double val);
    }

    public static interface ElementOpWithIndices {
        double op(int r, int c, double val);
    }

    public static double sigmoid(double num) {
        return 1 / (Math.exp(-num) + 1);
    }

    public static double sigmoidDerivative(double sigmoid) {
        return sigmoid * (1 - sigmoid);
    }

    public static SimpleMatrix elementOp(SimpleMatrix m, ElementOp elementOp) {
        return m.elementOp((int r, int c, double v) -> elementOp.op(v));
    }

    public static SimpleMatrix elementOp(SimpleMatrix m, ElementOpWithIndices elementOp) {
        return m.elementOp(elementOp::op);
    }

    public static SimpleMatrix indexWiseOp(SimpleMatrix m1, SimpleMatrix m2, ToDoubleBiFunction<Double, Double> fn) {
        SimpleMatrix o = new SimpleMatrix(m1.getNumRows(), m1.getNumCols());
        for (int r = 0; r < m1.getNumRows(); r++) {
            for (int c = 0; c < m1.getNumCols(); c++) {
                o.set(r, c, fn.applyAsDouble(m1.get(r, c), m2.get(r, c)));
            }
        }
        return o;
    }

    public static SimpleMatrix indexWiseOp(SimpleMatrix m, DoubleFunction<Double> fn) {
        SimpleMatrix o = new SimpleMatrix(m.getNumRows(), m.getNumCols());
        for (int r = 0; r < m.getNumRows(); r++) {
            for (int c = 0; c < m.getNumCols(); c++) {
                o.set(r, c, fn.apply(m.get(r, c)));
            }
        }
        return o;
    }

    public static SimpleMatrix random(int numRows, int numCols, Random random) {
        return SimpleMatrix.random_DDRM(numRows, numCols, -1, 1, random);
    }

    public static void exit() {
        System.exit(1);
    }

    public static void serialize(Object network, String filepath) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(filepath);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(network);
        objectOutputStream.flush();
        objectOutputStream.close();
    }


    public static Object deserialize(String filepath) throws ClassNotFoundException, IOException {
        FileInputStream fileInputStream = new FileInputStream(filepath);
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        Object o = objectInputStream.readObject();
        objectInputStream.close();
        return o;
    }
}
