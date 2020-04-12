package moa.classifiers.active;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;


/**
 * An adaptive streaming active learning strategy based on instance weighting.
 *
 * Implementation of the algorithm presented in the following publication:
 * Bouguelia, Mohamed-Rafik, Yolande Belaïd, and Abdel Belaïd. "An adaptive streaming active learning strategy based on
 * instance weighting." Pattern Recognition Letters 70 (2016): 38-44.
 */
public class ALInstanceWeighting extends AbstractClassifier implements ALClassifier {

    @Override
    public String getPurposeString() {
        return "An adaptive streaming active learning strategy based on instance weighting";
    }

    /**
     * Underlying learner.
     */
    public ClassOption baseLearnerOption = new ClassOption("baseLearner",
            'l', "Classifier to train.", Classifier.class, "bayes.NaiveBayes");

    /**
     * Learning rate, regulates the size of a step at which the confidence threshold changes.
     */
    public FloatOption learningRateOption = new FloatOption("learningRate",
            'r', "Learning rate.",
            0.001, 0.00001, 0.99);

    /**
     * Learning rate, regulates the size of a step at which the confidence threshold changes.
     */
    public FloatOption initialConfidenceThresholdOption = new FloatOption("initialConfidenceThreshold",
            'k', "Initial confidence threshold.",
            0.5, 0.000001, 0.99);

    /**
     * Number of instances that will be used for training the model at the start.
     */
    public FloatOption numInstancesInitOption = new FloatOption("numInstancesInit",
            'n', "Number of instances at beginning without active learning.",
            0.0, 0.00, Integer.MAX_VALUE);

    /**
     * If set to false, unlimited budget.
     */
    public FlagOption allowBudgetFlagOption = new FlagOption("allowBudget",
            'z', "Limit the budget");

    /**
     * Total budget.
     */
    public FloatOption budgetOption = new FloatOption("budget",
            'b', "Budget.",
            1.00, 0.0, 1.00);

    /**
     * Cost of requesting a label.
     */
    public FloatOption costOption = new FloatOption("cost",
            's', "Floating budget step.",
            0.01, 0.0, 1.00);

    /**
     * Number of processed instances after which the budget value is reset.
     */
    public IntOption budgetRestCount = new IntOption("budgetRestCount", 'p',
            "Number of processed instances after which the budget value is reset.",
            100, 1, Integer.MAX_VALUE);

    /**
     * Underlying classifier.
     */
    public Classifier classifier;

    private int _lastLabelAcq = 0;

    private int _iterationControl;

    private double _confidenceThreshold;

    private double _budgetSpent;

    private int _instanceProcessed;

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return this.classifier.getVotesForInstance(inst);
    }

    @Override
    public void resetLearningImpl() {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this._iterationControl = 0;
        this._confidenceThreshold = initialConfidenceThresholdOption.getValue();
        this._lastLabelAcq = 0;
        this._budgetSpent = 0;
        this._instanceProcessed = 0;
    }

    /**
     * Obtain the value of weight sufficient for swinging the models opinion on the instance
     * @param inst Instance in question
     * @param y1 Highest posterior label returned by the underlying model
     * @param y2 Second highest posteri_instanceProcessedor label returned by the underlying model
     * @return Weight value sufficient to swing the model's opinion
     */
    public double getSufficientWeight(Instance inst, int y1, int y2){
        double low = 0;
        double up = 1;
        double w;
        do{
            w = (up+low)/2;
            Classifier hCopy = classifier.copy();
            Instance xCopy = inst.copy();
            xCopy.setWeight(w);
            xCopy.setClassValue(y2);
            hCopy.trainOnInstance(xCopy);
            double[] p = hCopy.getVotesForInstance(inst);
            int maxPosteriorIndex = Utils.maxIndex(p);
            if(maxPosteriorIndex == y1){
                low = w;
            }
            else{
                up = w;
            }
        }while( (up - low) > learningRateOption.getValue());
        return w;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        this._iterationControl++;

        if (this._iterationControl <= this.numInstancesInitOption.getValue()) {
            this.classifier.trainOnInstance(inst);
            return;
        }

        double[] p = classifier.getVotesForInstance(inst);
        // Modification of the algorithm
        // if it's only aware of only one label it won't be able to proceed so we have to request a labeler
        if(p.length < 2){
            this._lastLabelAcq += 1;
            classifier.trainOnInstance(inst);
            return;
        }
        _instanceProcessed++;
        boolean withinBudget = isBudgetAllowingToProceed();
        if(!withinBudget){
            return;
        }

        int y1Index = Utils.maxIndex(p);
        double y1Posterior = p[y1Index];
        // If the predicted posterior is 0 then we need to train our model regardless
        if(y1Posterior == 0){
            this._lastLabelAcq += 1;
            classifier.trainOnInstance(inst);
            return;
        }

        p[y1Index] = -1;
        int y2Index = Utils.maxIndex(p);

        Classifier hCopy = classifier.copy();
        Instance xCopy = inst.copy();
        xCopy.setWeight(_confidenceThreshold);
        xCopy.setClassValue(y2Index);
        hCopy.trainOnInstance(xCopy);

        double[] pCopy = hCopy.getVotesForInstance(inst);
        int maxPosteriorIndex = Utils.maxIndex(pCopy);
        if(maxPosteriorIndex != y1Index){
            if(allowBudgetFlagOption.isSet()) {
                _budgetSpent += costOption.getValue();
            }
            int y = (int)inst.classValue();
            this._lastLabelAcq += 1;
            double w = getSufficientWeight(inst, y1Index, y2Index);
            double learningRate = learningRateOption.getValue();
            if( y == y1Index){
                _confidenceThreshold = _confidenceThreshold - learningRate * (_confidenceThreshold - w);
            }else{
                _confidenceThreshold = _confidenceThreshold + learningRate * ((w * (1 - _confidenceThreshold)) / _confidenceThreshold);
            }
            classifier.trainOnInstance(inst);
        }
    }

    /**
     * Test if there is enough budget left to proceed within t.
     * @return True - sufficient budget to request the labeller.
     */
    private boolean isBudgetAllowingToProceed(){
        if(!allowBudgetFlagOption.isSet()){
            return true;
        }

        if(_instanceProcessed >= budgetRestCount.getValue()){
            _instanceProcessed = 0;
            _budgetSpent = 0;
            return true;
        }

        double nextBudgetValue = _budgetSpent + costOption.getValue();
        return nextBudgetValue <= budgetOption.getValue();
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<>();
        measurementList.add(new Measurement("confidence threshold", this._confidenceThreshold));
        measurementList.add(new Measurement("in current window", this._instanceProcessed));
        measurementList.add(new Measurement("budget spent", this._budgetSpent));
        Measurement[] modelMeasurements = (this.classifier).getModelMeasurements();
        if (modelMeasurements != null) {
            Collections.addAll(measurementList, modelMeasurements);
        }
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }

    /**
     * Returns a string representation of the model.
     *
     * @param out The stringbuilder to add the description
     * @param indent The number of characters to indent
     */
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        ((AbstractClassifier) this.classifier).getModelDescription(out, indent);
    }

    /**
     * Returns true if the previously chosen instance was added to the training set
     * of the active learner.
     *
     */
    @Override
    public int getLastLabelAcqReport() {
        int help = this._lastLabelAcq;
        this._lastLabelAcq = 0;
        return help;
    }

    /**
     * Gets whether this learner needs a random seed.
     * Examples of methods that needs a random seed are bagging and boosting.
     *
     * @return true if the learner needs a random seed.
     */
    @Override
    public boolean isRandomizable() {
        return true;
    }
}
