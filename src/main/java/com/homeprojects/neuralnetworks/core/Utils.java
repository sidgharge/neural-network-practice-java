package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

import java.util.Random;

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

    public static SimpleMatrix random(int numRows, int numCols, Random random) {
        return SimpleMatrix.random_DDRM(numRows, numCols, -1, 1, random);
    }

    public static void exit() {
        System.exit(1);
    }

}
