package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

import java.util.Random;
import java.util.function.Function;

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

    public static void printDims(String name, SimpleMatrix matrix) {
        System.out.printf("%s = %d x %d \n", name, matrix.getNumRows(), matrix.getNumCols());
    }

    public static void printLine() {
        System.out.println("---------------------------------------");
    }
}
