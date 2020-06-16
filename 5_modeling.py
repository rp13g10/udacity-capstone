import pyspark.ml as sm
import pyspark.ml.classification as smc
import pyspark.ml.evaluation as sme
import pyspark.ml.feature as smf
import pyspark.ml.tuning as smt

import importlib
p2 = importlib.import_module(
    '2_load_clean_data')

spark = p2.spark
script_env = p2.script_env

# Set up pipeline ##################################################################################

encoded = spark.read.parquet('data/cached/encoded.parquet')
encoded_sample = encoded.limit(1000).toPandas()

# Use VectorAssembler to combine all features into a single vector
feature_cols = [x for x in encoded.columns if x not in {'userId', 'userChurnFlag'}]
assembler = smf.VectorAssembler(
    inputCols=feature_cols,
    outputCol='features')
encoded = assembler.transform(encoded)
encoded = encoded.drop(*feature_cols)
encoded = encoded.withColumnRenamed('userChurnFlag', 'label')

# Split out validation dataset
train, val = encoded.randomSplit([3.0, 1.0], seed=42)

# Set up pipeline for model training/evaluation
scaler = smf.StandardScaler(
    withStd=True,
    withMean=False,
    inputCol='features',
    outputCol='scaledFeatures')


# Use PCA to reduce dimensionality of scaled vectors
reducer = smf.PCA(
    k=10,
    inputCol=scaler.getOutputCol(),
    outputCol='selectedFeatures')

# Use a classifier to generate the final predictions
classifier = smc.GBTClassifier(
    labelCol='label',
    featuresCol=reducer.getOutputCol(),
    predictionCol='predictedLabel'
)

# Combine all steps in a pipeline
pipeline = sm.Pipeline(
    stages=[scaler, reducer, classifier]
)

# Create an evaluator which will quantify model performance
evaluator = sme.BinaryClassificationEvaluator(
    labelCol='label',
    rawPredictionCol='predictedLabel',
    metricName='areaUnderROC'
)

# Set up a parameter grid for cross validation
param_grid = smt.ParamGridBuilder().addGrid(
    reducer.k, [10, 20, 50, 75]
).addGrid(
    classifier.maxDepth, [2, 5, 10]
).addGrid(
    classifier.subsamplingRate, [0.1, 0.2, 0.3]
)

# Bring everything together
validator = smt.CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3
)

# Fit the model to the data #######################################################################'
model = validator.fit(train)

train_predictions = model.transform(train)
val_predictions = model.transform(val)