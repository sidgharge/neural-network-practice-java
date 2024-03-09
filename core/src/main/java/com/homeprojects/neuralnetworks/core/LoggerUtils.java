package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

public class LoggerUtils {

    public static boolean debug = false;

    public static void printLine() {
        if (!debug) {
            return;
        }
        System.out.println("---------------------------------------");
    }

    public static void printDims(String name, SimpleMatrix matrix) {
        if (!debug) {
            return;
        }
        System.out.printf("%s = %d x %d \n", name, matrix.getNumRows(), matrix.getNumCols());
    }

    public static void printLayerNumber(int i) {
        if (!debug) {
            return;
        }
        System.out.println("------------ Layer " + i + " ------------------");
    }

    public static void print(String name, SimpleMatrix matrix) {
        System.out.printf("%s:\n", name);
        for (int r = 0; r < matrix.getNumRows(); r++) {
            for (int c = 0; c < matrix.getNumCols(); c++) {
                System.out.printf("%9f\t", matrix.get(r, c));
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void print(SimpleMatrix matrix) {
        print("matrix", matrix);
    }
}
