package moa.classifiers.active;

import junit.framework.Test;
import junit.framework.TestSuite;
import moa.classifiers.AbstractMultipleClassifierTestCase;
import moa.classifiers.Classifier;

/**
 * Tests ALInstanceWeighting classifier.
 */
public class ALInstanceWeightingTest
        extends AbstractMultipleClassifierTestCase {

    public ALInstanceWeightingTest(String name){
        super(name);
        this.setNumberTests(1);
    }

    @Override
    protected Classifier[] getRegressionClassifierSetups() {
        return new Classifier[]{
            new ALInstanceWeighting()
        };
    }

    /**
     * Returns a test suite.
     *
     * @return		the test suite
     */
    public static Test suite() {
        return new TestSuite(ALRandomTest.class);
    }

    /**
     * Runs the test from commandline.
     *
     * @param args	ignored
     */
    public static void main(String[] args) {
        runTest(suite());
    }
}
