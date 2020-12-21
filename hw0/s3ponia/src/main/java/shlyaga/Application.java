package shlyaga;

import com.github.davidmoten.geo.GeoHash;
import com.github.davidmoten.geo.LatLong;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.expressions.Window;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.corr;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.first;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.udf;
import static org.apache.spark.sql.functions.variance;

import org.apache.spark.sql.types.DataTypes;

import java.net.URL;

public class Application {

    public static void main(String[] args) {

        final SparkSession spark = SparkSession
                .builder()
                .config("spark.master", "local")
                .getOrCreate();

        final Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("mode", "DROPMALFORMED")
                .option("escape", "\"")
                .option("quote", "\"")
                .csv("src/main/resources/AB_NYC_2019.csv");

        data.printSchema();

        data
                .withColumn("cnt", count("price")
                        .over(Window
                                .partitionBy("room_type")
                        )
                )
                .withColumn("price_mode", first("price").over(Window.orderBy("cnt").partitionBy("room_type")).as("mode"))
                .groupBy("room_type")
                .agg(
                        first("price_mode").as("mode"),
                        callUDF("percentile_approx", col("price"), lit(0.5)).as("median"),
                        avg("price").as("avg"),
                        variance("price").as("variance")
                )
                .show();
        /*
         * +---------------+-----+------+------------------+------------------+
         * |      room_type| mode|median|               avg|          variance|
         * +---------------+-----+------+------------------+------------------+
         * |    Shared room| 40.0|  45.0| 70.13298791018998|10365.890682680929|
         * |Entire home/apt|225.0| 160.0|211.88216032823104| 80852.24645965557|
         * |   Private room|149.0|  70.0| 89.51396823968689|23907.680804069663|
         * +---------------+-----+------+------------------+------------------+
         */

        data.orderBy("price").show(1);
        /*
         * +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         * |      id|                name|host_id|host_name|neighbourhood_group|     neighbourhood|latitude|longitude|   room_type|price|minimum_nights|number_of_reviews|last_review|reviews_per_month|calculated_host_listings_count|availability_365|
         * +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         * |18750597|Huge Brooklyn Bro...|8993084| Kimberly|           Brooklyn|Bedford-Stuyvesant|40.69023|-73.95428|Private room|    0|             4|                1| 2018-01-06|             0.05|                           4.0|              28|
         * +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         */

        data.orderBy(desc("price")).show(1);
        /*
         * +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         * |     id|                name|host_id|host_name|neighbourhood_group|  neighbourhood|latitude|longitude|   room_type|price|minimum_nights|number_of_reviews|last_review|reviews_per_month|calculated_host_listings_count|availability_365|
         * +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         * |9528920|Quiet, Clean, Lit...|3906464|      Amy|          Manhattan|Lower East Side|40.71355|-73.98507|Private room| 9999|            99|                6| 2016-01-01|             0.14|                           1.0|              83|
         * +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
         */

        data
                .agg(
                        corr("price", "minimum_nights").as("price_min_nights"),
                        corr("price", "number_of_reviews").as("price_num_of_reviews")
                )
                .show();
        /*
         * +-------------------+--------------------+
         * |   price_min_nights|price_num_of_reviews|
         * +-------------------+--------------------+
         * |0.04238800501413225|-0.04806955416645...|
         * +-------------------+--------------------+
         */

        final UserDefinedFunction geoHash = udf(
                (UDF3<Double, Double, Integer, String>) GeoHash::encodeHash,
                DataTypes.StringType
        );

        final UserDefinedFunction geoHashToLatitude = udf(
                (UDF1<String, Double>) hash ->
                        GeoHash.decodeHash(hash).getLat(), DataTypes.DoubleType
        );

        final UserDefinedFunction geoHashToLongitude = udf(
                (UDF1<String, Double>) hash ->
                        GeoHash.decodeHash(hash).getLon(), DataTypes.DoubleType
        );
        data
                .withColumn("geoHash", geoHash.apply(
                        col("latitude").cast(DataTypes.DoubleType),
                        col("longitude").cast(DataTypes.DoubleType),
                        lit(5)
                ))
                .groupBy("geoHash")
                .agg(
                        avg("price").as("avg_price")
                )
                .select(
                        geoHashToLatitude.apply(col("geoHash")).as("Latitude"),
                        geoHashToLongitude.apply(col("geoHash")).as("Longitude"),
                        col("avg_price").as("Average price")
                )
                .orderBy(desc("Average price"))
                .show(1);
        /*
         * +--------------+---------------+---------+
         * |      Latitude|      Longitude|avg_price|
         * +--------------+---------------+---------+
         * |40.58349609375|-73.71826171875|    350.0|
         * +--------------+---------------+---------+
         */
    }
}

