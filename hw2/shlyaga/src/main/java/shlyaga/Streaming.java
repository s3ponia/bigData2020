package shlyaga;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQueryException;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.from_json;

public class Streaming {
    public static void main(String[] args) throws StreamingQueryException {
        final SparkSession spark = SparkSession
                .builder()
                .config("spark.master", "local")
                .getOrCreate();

        final StructType structType = new StructType()
                .add("id", DataTypes.StringType, true)
                .add("text", DataTypes.StringType, true);

        final PipelineModel model = PipelineModel
                .read()
                .load("model/");

        final Dataset<Row> inputData = spark
                .readStream()
                .format("socket")
                .option("host", "localhost")
                .option("port", 9999)
                .load();

        final Dataset<Row> inputJson =
                inputData
                        .withColumn("json", from_json(col("value"), structType))
                        .select("json.*")
                        .select(col("id"), col("text"))
                        .filter(col("id").isNotNull())
                        .filter(col("text").isNotNull());

        inputJson.printSchema();
        model
                .transform(inputJson)
                .select(col("id"), col("text"), col("target"))
                .repartition(1)
                .writeStream()
                .outputMode("append")
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .option("path", "src/main/resources/out/")
                .option("checkpointLocation", "src/main/resources/checkpointLocation/")
                .start()
                .awaitTermination();
    }
}
