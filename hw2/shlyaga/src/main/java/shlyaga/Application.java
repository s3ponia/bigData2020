package shlyaga;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.feature.Stemmer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.io.IOException;
import java.util.ArrayList;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.when;

public class Application {

    public static void main(String[] args) throws IOException {
        final SparkSession spark = SparkSession
                .builder()
                .config("spark.master", "local")
                .getOrCreate();

        final Dataset<Row> train = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/train.csv")
                .filter(col("text").isNotNull())
                .filter(col("target").isNotNull())
                .select("id", "text", "target")
                .withColumnRenamed("target", "label");

        final Dataset<Row> testData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/test.csv")
                .filter(col("text").isNotNull())
                .filter(col("id").isNotNull())
                .select("id", "text");

        final Dataset<Row> sample = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/sample_submission.csv")
                .select("id");

        final RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("text")
                .setOutputCol("words")
                .setPattern("[\\W]");

        final StopWordsRemover wordsRemover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered_words");

        final Stemmer stemmer = new Stemmer()
                .setInputCol("filtered_words")
                .setOutputCol("stemmed")
                .setLanguage("English");

        final HashingTF hashingTF = new HashingTF()
                .setNumFeatures(3000)
                .setInputCol("stemmed")
                .setOutputCol("rowFeatures");

        final IDF idf = new IDF()
                .setInputCol(hashingTF.getOutputCol())
                .setOutputCol("features");

        final StringIndexer stringIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexed");

        final GBTClassifier gbt = new GBTClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setPredictionCol("target")
                .setMaxIter(30);

        final ArrayList<PipelineStage> stages = new ArrayList<PipelineStage>();
        stages.add(tokenizer);
        stages.add(wordsRemover);
        stages.add(stemmer);
        stages.add(hashingTF);
        stages.add(idf);
        stages.add(stringIndexer);
        stages.add(gbt);

        final Pipeline pipeline = new Pipeline()
                .setStages(
                        stages.toArray(
                                new PipelineStage[]{
                                        tokenizer,
                                        wordsRemover,
                                        stemmer,
                                        hashingTF,
                                        idf,
                                        stringIndexer,
                                        gbt
                                }
                        )
                );

        final PipelineModel pipelineModel = pipeline.fit(train);
        final Dataset<Row> pipelinedTest = pipelineModel.transform(testData);

        final Dataset<Row> result = pipelinedTest
                .select(
                        col("id"),
                        col("target")
                                .cast(DataTypes.IntegerType)
                );

        final Dataset<Row> joinedWithSample = result
                .join(sample,
                        sample
                                .col("id")
                                .equalTo(result.col("id")),
                        "right")
                .select(sample.col("id"),
                        (
                                when(
                                        result.col("id")
                                                .isNull(),
                                        lit(0)
                                ).otherwise(
                                        col("target")
                                )
                        ).as("target")
                );

        joinedWithSample.write()
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .save("src/main/resources/result.csv");

        pipelineModel
                .write()
                .overwrite()
                .save("model/");
    }
}

