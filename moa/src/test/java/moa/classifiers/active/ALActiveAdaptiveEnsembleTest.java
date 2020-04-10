package moa.classifiers.active;

import junit.framework.Test;
import junit.framework.TestSuite;
import moa.classifiers.AbstractMultipleClassifierTestCase;
import moa.classifiers.Classifier;

/**
 * Tests the ALActiveAdaptiveEnsemble classifier.
 */
public class ALActiveAdaptiveEnsembleTest
        extends AbstractMultipleClassifierTestCase {

    /**
     * Constructs the test case. Called by subclasses.
     *
     * @param name 	the name of the test
     */
    public ALActiveAdaptiveEnsembleTest(String name){
        super(name);
        this.setNumberTests(1);
    }

    /**
     * Returns the classifier setups to use in the regression test.
     *
     * @return		the setups
     */
    @Override
    protected Classifier[] getRegressionClassifierSetups() {
        return new Classifier[]{
                new ALActiveAdaptiveEnsemble()
        };
    }

    /**
     * Returns a test suite.
     *
     * @return		the test suite
     */
    public static Test suite() {
        return new TestSuite(ALActiveAdaptiveEnsemble.class);
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
