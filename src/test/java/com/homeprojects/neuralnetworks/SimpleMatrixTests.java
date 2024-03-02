package com.homeprojects.neuralnetworks;

import com.homeprojects.neuralnetworks.core.Utils;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class SimpleMatrixTests {

    @Test
    public void matrixFunction() {
        SimpleMatrix inputs = new SimpleMatrix(new double[][] {
                new double[] {1, 2, 3},
                new double[] {10, 20, 30}
        });

        SimpleOperations.ElementOpReal op = (r, c, n) -> n * 2;

        inputs.elementOp(op).print();
    }

    @Test
    public void sigmoid() {
        Assertions.assertEquals(BigDecimal.valueOf(Utils.sigmoid(5)).setScale(14, RoundingMode.HALF_UP)
                .compareTo(BigDecimal.valueOf(0.99330714907572)), 0);

        Assertions.assertEquals(BigDecimal.valueOf(Utils.sigmoid(0.12)).setScale(14, RoundingMode.HALF_UP)
                .compareTo(BigDecimal.valueOf(0.52996405176457)), 0);

        Assertions.assertEquals(BigDecimal.valueOf(Utils.sigmoid(-0.3)).setScale(14, RoundingMode.HALF_UP)
                .compareTo(BigDecimal.valueOf(0.42555748318834)), 0);
    }

}
