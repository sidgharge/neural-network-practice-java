package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

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

}
