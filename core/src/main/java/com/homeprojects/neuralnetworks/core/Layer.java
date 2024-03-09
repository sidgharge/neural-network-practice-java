package com.homeprojects.neuralnetworks.core;

import org.ejml.simple.SimpleMatrix;

import java.io.Serializable;

public record Layer(SimpleMatrix w, SimpleMatrix b, SimpleMatrix a) implements Serializable {

    public void print() {
        System.out.println("w = " + w);
        System.out.println("b = " + b);
        System.out.println("a = " + a);
    }
}
