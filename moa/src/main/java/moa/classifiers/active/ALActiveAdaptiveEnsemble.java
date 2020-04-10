package moa.classifiers.active;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Active and adaptive ensemble learning for online activity recognition from data streams.
 *
 * Implementation of the algorithm presented in the following publication:
 * Krawczyk, Bartosz. "Active and adaptive ensemble learning for online activity recognition from data streams."
 * Knowledge-Based Systems 138 (2017): 69-78.
 */
public class ALActiveAdaptiveEnsemble extends AbstractClassifier implements ALClassifier {
    private int _lastLabelAcq = 0;
    @Override
    public String getPurposeString() {
        return "Active and adaptive ensemble learning for online activity recognition from data streams";
    }

    /**
     * Base classifier.
     */
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "bayes.NaiveBayes");

    /**
     * Weight adjustment step.
     */
    public FloatOption weightAdjustmentOption = new FloatOption("weightStep",
            'd', "Weight adjustment step.",
            1.10, 0.0001, 100.00);

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
     * Window weight threshold.
     */
    public FloatOption windowWeightThreshold = new FloatOption("windowWeightThreshold",
            'j', "Window weight below which it will be discarded.",
            0.1, 0.0001, 0.9);

    /**
     * Confidence threshold initial value.
     */
    public FloatOption confidenceThresholdInitial = new FloatOption("confidenceThresholdInitial",
            'g', "Confidence threshold adjustment step.",
            0.75, 0.000001, 1);
    /**
     * Confidence threshold adjustment step.
     */
    public FloatOption thresholdAdjustmentStep = new FloatOption("thresholdAdjustmentStep",
            'n', "Confidence threshold adjustment step.",
            0.05, 0.000001, 0.99);
    /**
     * Forgetting speed.
     */
    public FloatOption forgettingSpeedOption = new FloatOption("forgettingSpeed",
            'f', "Forgetting speed.",
            0.9, 0.0001, 1);
    /**
     * Window size.
     */
    public IntOption windowSizeOption = new IntOption("windowSize",
            'w', "Size of the window (# of elements in the batch).",
            100, 1, Integer.MAX_VALUE);

    /**
     * Ensemble of classifiers currently in use
     */
    protected EnsembleSet ensembleSet = new EnsembleSet();

    /**
     * Current uncertainty threshold value
     */
    protected double uncertaintyThreshold = confidenceThresholdInitial.getValue();

    /**
     * Current active windows (sorted)
     */
    protected TreeSet<InstanceWindow> currentWindows = new TreeSet<>();

    @Override
    public double[] getVotesForInstance(Instance inst) {
        List<EnsembleClassificationResult> votes = classifyInstanceWithEnsemble(inst);
        DoubleVector result = new DoubleVector();
        for(EnsembleClassificationResult vote: votes){
            if(result.numValues() >= vote.classIndex){
                if(result.getValue(vote.classIndex) < vote.posterior){
                    result.setValue(vote.classIndex, vote.posterior);
                }
            }
            else{
                result.setValue(vote.classIndex, vote.posterior);
            }
        }
        return result.getArrayCopy();
    }

    @Override
    public void resetLearningImpl() {
        ensembleSet.clear();
        currentWindows.clear();
        uncertaintyThreshold = confidenceThresholdInitial.getValue();
        _tmpFirstDiscoveredInstanceClass = null;
        _lastLabelAcq = 0;
        _budgetSpent= 0;
    }

    private Instance _tmpFirstDiscoveredInstanceClass = null;
    private double _budgetSpent;
    /**
     * Current window where we're placing incoming instances
     */
    private InstanceWindow _currentWindow;

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if(_currentWindow == null){
            _currentWindow = new InstanceWindow();
        }
        Instance instanceCopy = inst.copy();
        _currentWindow.add(instanceCopy);
        // If windows is not yet full, terminate
        if(_currentWindow.size() < windowSizeOption.getValue()){
            return;
        }
        // Window of items that were used for training
        InstanceWindow trainingWindow = new InstanceWindow();

        // This initialization step isn't part of the actual algorithm, but we need to get at least one binary classifier
        // before we can processed with the actual algorithm.
        int windowStartingPoint = 0;
        if(ensembleSet.size() == 0){
            boolean secondLabelDiscovered = false;
            if(_tmpFirstDiscoveredInstanceClass == null) {
                // If the are pre-existing windows already grab the first class from there
                if(!currentWindows.isEmpty()){
                    if(currentWindows.last().elements().size() > 0){
                        _tmpFirstDiscoveredInstanceClass = currentWindows.last().elements().get(0).copy();
                        _tmpFirstDiscoveredInstanceClass.setWeight(1);
                    }
                }
            }

            for(windowStartingPoint = 0; windowStartingPoint < _currentWindow.size(); windowStartingPoint++){
                _lastLabelAcq++; // "request labeler"
                Instance cWindowInstance = _currentWindow.getInstance(windowStartingPoint);
                // Assign instance with the first discovered
                if(_tmpFirstDiscoveredInstanceClass == null){
                    _tmpFirstDiscoveredInstanceClass = cWindowInstance;
                    continue;
                }
                double thisClassValue = cWindowInstance.classValue();
                if(thisClassValue != _tmpFirstDiscoveredInstanceClass.classValue()){
                    secondLabelDiscovered = true;
                }
                trainingWindow.add(cWindowInstance);
                trainingWindow.add(_tmpFirstDiscoveredInstanceClass);
                _tmpFirstDiscoveredInstanceClass = null;
                if (secondLabelDiscovered) {
                    break;
                }
            }

            boolean ensembleInitialized = initializeEnsemble(trainingWindow);
            if (!ensembleInitialized) {
                // If we were unable to initialize the  ensemble with the instances we have, wait for more instances
                return;
            }
        }

        double thresholdAdjustment = thresholdAdjustmentStep.getValue();
        for(int i = windowStartingPoint; i < _currentWindow.size(); i++) {
            Instance cWindowInstance = _currentWindow.getInstance(i);
            // Check if there is sufficient budget
            if(this.budgetOption.getValue() > _budgetSpent){
                List<EnsembleClassificationResult> votes = classifyInstanceWithEnsemble(cWindowInstance);
                // Get class predicted by the ensemble
                EnsembleClassificationResult prediction = null;
                for (EnsembleClassificationResult vote : votes) {
                    if(prediction == null){
                        prediction = vote;
                    }
                    else if(vote.posterior > prediction.posterior){
                        prediction = vote;
                    }
                }
                // Get random multiplier
                double random = ThreadLocalRandom.current().nextDouble(0.01, 1);
                // Get randomized uncertainty threshold
                double randUT = this.uncertaintyThreshold * random;
                // Check if posterior of the model is < that the uncertainty threshold
                if(prediction.posterior < randUT){
                    // Request labeler
                    _budgetSpent += costOption.getValue();
                    double groundTruthLabelIndex = cWindowInstance.classValue();
                    this._lastLabelAcq += 1;
                    // Construct instance with ground truth
                    Instance trueInstance = cWindowInstance.copy();
                    trueInstance.setWeight(1.0);
                    trainingWindow.add(trueInstance);
                    // Decrease the uncertainty threshold
                    uncertaintyThreshold = adjustThresholdValue(uncertaintyThreshold, -1 * thresholdAdjustment);
                    // Updated classifiers
                    final double weightAdjustmentStep = weightAdjustmentOption.getValue();
                    for (EnsembleClassificationResult vote : votes) { // updated classifier weights
                        if(vote.classIndex == groundTruthLabelIndex){
                            // increase classifier weight
                            vote.decidingClassifier.setWeight(vote.decidingClassifier.getWeight() * weightAdjustmentStep);
                        }
                        else{
                            // reduce classifier weight
                            vote.decidingClassifier.setWeight(vote.decidingClassifier.getWeight() / weightAdjustmentStep);
                        }
                    }
                    trainEnsembleWithInstance(trueInstance);
                }
                else{
                    // Increase the uncertainty threshold
                    uncertaintyThreshold = adjustThresholdValue(uncertaintyThreshold, thresholdAdjustment);
                }
            }
        }

        currentWindows.add(trainingWindow);
        _currentWindow = null;

        decayProcedure(windowWeightThreshold.getValue());
        Iterator<InstanceWindow> it = currentWindows.iterator();
        ensembleSet.clear();
        if(initializeEnsemble(null)) {
            while (it.hasNext()) {
                InstanceWindow currentWindow = it.next();
                for (int i = 0; i < currentWindow.size(); i++) {
                    Instance cWindowInstance = currentWindow.getInstance(i);
                    cWindowInstance.setWeight(currentWindow.getCurrentWeight());
                    trainEnsembleWithInstance(cWindowInstance);
                }
            }
        }
        _budgetSpent = 0;
    }

    /**
     * Re-initialize the ensemble according to the currently present windows.
     */
    private boolean initializeEnsemble(InstanceWindow additionalWindow){
        Instance y1instance = null;
        Instance y2instance = null;
        for(InstanceWindow window: currentWindows){
            for (Instance inst: window.elements()){
                if(y1instance == null){
                    y1instance = inst.copy();
                    inst.setWeight(window.getCurrentWeight());
                } else if(y1instance.classValue() != inst.classValue()){
                    y2instance = inst.copy();
                    y2instance.setWeight(window.getCurrentWeight());
                    break;
                }
            }
            if(y1instance != null && y2instance != null){
                break;
            }
        }
        if(y1instance == null || y2instance == null){
            if(additionalWindow != null){
                for (Instance inst: additionalWindow.elements()){
                    if(y1instance == null){
                        y1instance = inst.copy();
                        inst.setWeight(1);
                    } else if(y1instance.classValue() != inst.classValue()){
                        y2instance = inst.copy();
                        y2instance.setWeight(1);
                        break;
                    }
                }
            }
            if(y1instance == null || y2instance == null){
                return false;
            }
        }
        Classifier newClassifier =  ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        newClassifier.resetLearning();
        EnsembleMember newEnsembleMember = new EnsembleMember((int) y1instance.classValue(),
                (int) y2instance.classValue(),
                newClassifier);
        newEnsembleMember.classifier.trainOnInstance(y1instance);
        newEnsembleMember.classifier.trainOnInstance(y2instance);
        ensembleSet.add(newEnsembleMember);

        return true;
    }

    /**
     * Calculate a new value of confidence threshold
     * @param currentValue Current threshold value
     * @param thresholdAdjustment Positive or negative threshold adjustment
     * @return New threshold value
     */
    private double adjustThresholdValue(double currentValue, double thresholdAdjustment){
        double result = currentValue + thresholdAdjustment;
        // Restrict threshold to the [0,1] range
        if(result > 1.0){
            return 1.0;
        } else if (result < 0.0){
            return 0.0;
        }
        return result;
    }

    /**
     * Classify instance over the current ensemble
     * @param inst Data instance
     * @return Classification results for all ensemble members
     */
    protected List<EnsembleClassificationResult> classifyInstanceWithEnsemble(Instance inst){
        ArrayList<EnsembleClassificationResult> votes = new ArrayList<>();
        for (EnsembleMember currentEnsembleMember : ensembleSet) {
            double[] votesForInstance = currentEnsembleMember.classifier.getVotesForInstance(inst);
            if (votesForInstance.length > 0) {
                int classIndex = Utils.maxIndex(votesForInstance);
                double weightedPosterior = votesForInstance[classIndex] * currentEnsembleMember.getWeight();
                EnsembleClassificationResult newResult
                        = new EnsembleClassificationResult(classIndex, weightedPosterior, currentEnsembleMember);
                votes.add(newResult);
            }
        }

        return votes;
    }

    /**
     * Train ensemble with the instance
     * @param inst Instance with a correct class label
     */
    protected void trainEnsembleWithInstance(Instance inst){
        Set<Double> exitingLabels = ensembleSet.getKnownLabels();
        List<EnsembleMember> ensembleMembers;
        if(!exitingLabels.contains(inst.classValue())){
            Classifier classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
            ensembleMembers = ensembleSet.generateEnsembleCombinationsForLabel(inst.classValue(),classifier);
        }
        else{
            ensembleMembers = ensembleSet.getEnsembleMembersForLabel(inst.classValue());
        }

        for(EnsembleMember currentMember: ensembleMembers){
            currentMember.classifier.trainOnInstance(inst);
        }
    }

    /**
     * Reduce weights for all of the previously remembered windows proportionally (except the last one added)
     * @param discardThreshold Minimum weight threshold below which windows should be removed
     */
    protected void decayProcedure(double discardThreshold){
        Iterator<InstanceWindow> it = currentWindows.descendingIterator();
        ArrayList<InstanceWindow> deleteBuffer = null;
        //it.next(); // Skip the top window
        int k = 1;
        while(it.hasNext()){
            InstanceWindow currentWindow = it.next();
            double w = currentWindow.getCurrentWeight();
            double bt = forgettingSpeedOption.getValue();

            double expTerm = Math.exp(-1.0 * bt * (k + 1.0));
            double newW = Math.pow(Math.pow(w, k) * 2 * expTerm * (1 + expTerm), 1.0/(k+1.0));
            currentWindow.setCurrentWeight(newW > 1? 1 : newW);
            k++;
            // Check if it should be discarded
            if(currentWindow.getCurrentWeight() < discardThreshold){
                if(deleteBuffer == null){
                    deleteBuffer = new ArrayList<>();
                }
                deleteBuffer.add(currentWindow);
            }
        }
        // Discard elements with too small weight
        if(deleteBuffer != null && deleteBuffer.size() > 0){
            currentWindows.removeAll(deleteBuffer);
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<>();
        measurementList.add(new Measurement("Uncertainty threshold", this.uncertaintyThreshold));
        measurementList.add(new Measurement("Budget Spent", this._budgetSpent));
        measurementList.add(new Measurement("Num. of windows", this.currentWindows.size()));
        measurementList.add(new Measurement("Num. ensemble members", this.ensembleSet.size()));
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO
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
        return false;
    }

    /**
     * Custom collection type for the ensemble members.
     * Implies indexed order, doesn't allow duplicate ensemble members
     */
    private static class EnsembleSet extends AbstractList<EnsembleMember> implements Serializable {
        /**
         * List of all members of this ensemble
         */
        private final ArrayList<EnsembleMember> _list;
        /**
         * Default constructor
         */
        public EnsembleSet() {
            _list = new ArrayList<>();
        }

        public Iterator<EnsembleMember> iterator() {
            return _list.iterator();
        }

        public int size() {
            return _list.size();
        }

        @Override
        public boolean add(EnsembleMember element) {
            // Prevent addition of the ensemble members classifying over the same pair of labels
            for (EnsembleMember currentMember: _list) {
                if(element.compareTo(currentMember) == 0){
                    return false;
                }
            }
            _list.add(element);
            return true;
        }

        public boolean remove(EnsembleMember element) {
            return _list.remove(element);
        }

        @Override
        public boolean isEmpty() {
            return _list.isEmpty();
        }

        public boolean contains(EnsembleMember element) {
            return _list.contains(element);
        }
        @Override
        public void clear() {
            _list.clear();
        }

        public EnsembleMember get(int index){
            return _list.get(index);
        }

        /**
         * Get list of all known to ensemble labels.
         * @return Set of all labels known by the ensemble.
         */
        public Set<Double> getKnownLabels(){
            TreeSet<Double> result = new TreeSet<>();
            for(EnsembleMember currentMember: this._list){
                result.add(currentMember.a);
                result.add(currentMember.b);
            }
            return result;
        }

        /**
         * Remove all ensemble members for a specific label.
         * @param label Label.
         */
        public void removeAllForLabel(int label){
            this._list.removeIf(m -> m.a == label || m.b == label);
        }

        /**
         * Reset all of the classifiers in the ensemble without removing them
         */
        public void resetClassifiers(){
            for(EnsembleMember currentMember: this._list){
                currentMember.classifier.resetLearning();
            }
        }

        /**
         * Retrieve all ensemble members specific for the label.
         * @param label Label index.
         * @return List of ensemble members related to the label.
         */
        public List<EnsembleMember> getEnsembleMembersForLabel(double label){
            ArrayList<EnsembleMember> result = new ArrayList<>();
            for(EnsembleMember existingMember: _list){
                if(existingMember.a == label || existingMember.b == label){
                    result.add(existingMember);
                }
            }
            return result;
        }

        /**
         * Generate all possible combinations between a new label and an exiting labels
         * @param label Label index
         * @param baseClassifier Base classifier (will be used as a template for a new ensemble member)
         * @return list of the newly created ensemble members
         */
        public List<EnsembleMember> generateEnsembleCombinationsForLabel(double label, Classifier baseClassifier){
            Set<Double> exitingLabels = getKnownLabels();
            if(exitingLabels.contains(label)){
                throw new IllegalArgumentException(String.format("Label %4.3f already exists in the ensemble", label));
            }

            ArrayList<EnsembleMember> result = new ArrayList<>();
            for(Double existingLabel: exitingLabels){
                Classifier classifier = baseClassifier.copy();
                classifier.resetLearning();

                EnsembleMember newEnsembleMember = new EnsembleMember(label, existingLabel, classifier);
                _list.add(newEnsembleMember);
                result.add(newEnsembleMember);
            }

            return result;
        }
    }

    /**
     * Class container for the classifier members
     */
    protected static class EnsembleMember implements Comparable<EnsembleMember>{
        /**
         * Label A index
         */
        public final double a;
        /**
         * Label B index
         */
        public final double b;
        /**
         * Underlying classifier
         */
        public final Classifier classifier;
        /**
         * Classifier weight
         */
        protected double weight = 1; // start with 1

        /**
         * Set weight for the ensemble member.
         * @param weight New instance weight value that ∈ [0,1].
         */
        public void setWeight(double weight) {
            if(weight <= 0){
                throw new IllegalArgumentException("Weight of the classifier can't be less or equal 0");
            }
            this.weight = weight;
        }

        /**
         * Get current weight of the ensemble member
         * @return Instance weight ∈ [0,1].
         */
        public double getWeight() {
            return weight;
        }

        /**
         * Default constructor
         * @param a Label A for the binary classifier
         * @param b Label B for the binary classifier
         * @param classifier Actual classifier implementation
         */
        public EnsembleMember(double a, double b, Classifier classifier) {
            this.a = a;
            this.b = b;
            this.classifier = classifier;
        }

        @Override
        public int compareTo(EnsembleMember o) {
            if(o == null){
                throw new IllegalArgumentException("can't compare to NULL");
            }
            // if two ensemble members classify over the same pair of attributes, they are considered equal
            if((this.a == o.a && this.b == o.b) || (this.a == o.b && this.b == o.a)){
                return 0;
            }
            // Order doesn't really matter in this case
            return -1;
        }
    }

    /**
     * Window container
     */
    protected static class InstanceWindow implements Comparable<InstanceWindow>{
        /**
         * Weight of the window. Correspond to the weights of all instances within this window.
         */
        private double _currentWeight = 1;
        /**
         * Items stored in the window.
         */
        private final ArrayList<Instance> _instances;
        /**
         * Timestamp when the window was created (epoch nanoseconds).
         */
        protected final long timestamp;

        /**
         * Get current weight.
         * @return value in [0,1].
         */
        double getCurrentWeight(){
            return _currentWeight;
        }

        /**
         * Assign new weight value for all instances in this window.
         * @param value value in [0,1].
         */
        void setCurrentWeight(double value){
            if(value < 0){
                throw new IllegalArgumentException("Weight can't be negative");
            }
            _currentWeight = value;
            // Update all of the instances in the window
            for(Instance instance: this._instances){
                instance.setWeight(value);
            }
        }

        /**
         * Default constructor.
         */
        public InstanceWindow() {
            this._instances = new ArrayList<>();
            this.timestamp = System.nanoTime();
        }

        @Override
        public int compareTo(InstanceWindow o) {
            return Long.compare(timestamp, o.timestamp);
        }

        /**
         * Add new instance to the window.
         * @param instance Instance object.
         */
        public synchronized void add(Instance instance){
            Objects.requireNonNull(instance, "Unable to add null to the window of instances");
            _instances.add(instance);
            if(instance.weight() != this.getCurrentWeight()){
                instance.setWeight(this.getCurrentWeight());
            }
        }

        /**
         * Get number of windows in the buffer.
         * @return Number of windows currently in the buffer.
         */
        public int size(){
            return _instances.size();
        }

        /**
         * Get instance object in specific index.
         * @param index Index of the instance in the list.
         * @return Instance for the index.
         */
        public Instance getInstance(int index){
            return this._instances.get(index);
        }

        /**
         * Get read only elements list view.
         * @return Get the list of all instances currently in the window.
         */
        public List<Instance> elements(){
            return Collections.unmodifiableList(_instances);
        }
    }

    /**
     * Class container used as a return value from the ensemble of classifiers.
     * Contains all of the information required for the correct work of the algorithm.
     */
    protected static final class EnsembleClassificationResult{
        /**
         * Predicted class index.
         */
        public final int classIndex;
        /**
         * Index of the classifier which decision was used.
         */
        public final EnsembleMember decidingClassifier;
        /**
         * Posterior value.
         */
        public final double posterior;

        /**
         * Default constructor.
         * @param classIndex Classification class index
         * @param decidingClassifierIndex Index of the classifier which decision was used
         * @param posterior Posterior value
         */
        public EnsembleClassificationResult(int classIndex, double posterior, EnsembleMember decidingClassifierIndex) {
            this.classIndex = classIndex;
            this.decidingClassifier = decidingClassifierIndex;
            this.posterior = posterior;
        }
    }
}