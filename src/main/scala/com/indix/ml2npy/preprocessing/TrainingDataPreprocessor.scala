package com.indix.ml2npy.preprocessing

import com.indix.ml2npy.hadoop.NpyOutPutFormat
import com.indix.ml2npy.text.CooccurrenceTokenizer
import org.apache.spark.SparkException
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}

import scala.util._

case class TrainingRecord(storeId: Long,
                          url: String, title: String, breadCrumbs: String, brandText: String, categoryPath: String, price: Double, isbn: String, asin: String, specificationText: String, leafId: String,
                          topLevelId: String, isBroken: Boolean, discoveredDate: Option[Long], discontinuedDate: Option[Long])

case class SimpleTrainingRecord(url: String, doc: String, topLevelId: Short, categoryId: Short)

case class TFIdfModel(vocab: Array[String], idf: Option[Vector])

trait DocGenerator {

  def main(args: Array[String]): Unit = {

    val inputPath: String = args(0)
    val outputPath: String = args(1)
    val numParts: Int = args(2).toInt

    val spark = SparkSession.builder().appName("Ml2Npy").getOrCreate()
    prepare(spark, inputPath, outputPath, numParts)
  }

  def tokenizer: PipelineStage

  def prepare(spark: SparkSession, inputPath: String, outputPath: String, numParts: Int): Unit = {
    import spark.implicits._
    val trainingRecords: Dataset[TrainingRecord] = spark.read.json(inputPath).as[TrainingRecord].cache()
    val toplevelTrainingRecords = trainingRecords.mapPartitions(iterator => {
      val bcSelector = new Random(11)
      val brandSelector = new Random(17)
      for {
        record <- iterator
      } yield {
        val isBc = bcSelector.nextInt(2) == 0
        val isBrand = brandSelector.nextInt(2) == 0
        val isSpecs = bcSelector.nextInt(2) == 0
        val doc = {
          record.title + {
            if (isBrand) " " + record.brandText else " "
          } + {
            if (isBc) " " + record.breadCrumbs else " "
          }
        }
        SimpleTrainingRecord(record.url, doc, Integer.parseInt(record.topLevelId).toShort, Integer.parseInt(record.leafId).toShort)
      }
    })

    val docTokenizer = tokenizer

    val docCV = new CountVectorizer()
      .setMinDF(100)
      .setInputCol("tokens")
      .setOutputCol("tokenCounts")
      .setBinary(true)

    val docIDF = new IDF()
      .setInputCol("tokenCounts")
      .setOutputCol("tokenIDF")

    val normalizer = new Normalizer()
      .setInputCol("tokenIDF")
      .setOutputCol("normalizedTokenVector")

    val pipeline = new Pipeline()
      .setStages(Array(docTokenizer, docCV, docIDF, normalizer))

    val ppModel = pipeline.fit(toplevelTrainingRecords)
    val transformedTopData = ppModel.transform(toplevelTrainingRecords)

    transformedTopData.select("doc", "tokens", "normalizedTokenVector").show()

    val idfData = transformedTopData.select("normalizedTokenVector", "topLevelId", "categoryId")
      .map(x => (x.getAs[Vector](0), new DenseVector(Array(x.getShort(1), x.getShort(2)))))
      .repartition(numParts)
      .rdd
      .saveAsHadoopFile(outputPath, classOf[Vector], classOf[Vector], classOf[NpyOutPutFormat])

    val cvModel = ppModel.stages(1) match {
      case x: CountVectorizerModel => x
    }
    val idfModel = ppModel.stages(2) match {
      case x: IDFModel => x
    }
    val tfidf: TFIdfModel = TFIdfModel(cvModel.vocabulary, Some(idfModel.idf))
    val data: Seq[TFIdfModel] = Seq(tfidf)
    val tfidfDF = spark.createDataFrame[TFIdfModel](data).coalesce(1).toDF()
    tfidfDF.write.json(s"$outputPath/tfidfModel")
  }
}

/**
  * Created by vumaasha on 28/12/16.
  */
object UnigramTokens extends DocGenerator {

  override def tokenizer: PipelineStage = {
    val docTokenizer = new RegexTokenizer()
      .setGaps(false)
      .setMinTokenLength(3)
      .setToLowercase(true)
      .setPattern("\\b[a-z0-9]+")
      .setInputCol("doc")
      .setOutputCol("tokens")
    docTokenizer
  }
}

object CooccTokens extends DocGenerator {

  override def tokenizer: PipelineStage = {
    val docTokenizer = new CooccurrenceTokenizer()
      .setGaps(false)
      .setMinTokenLength(3)
      .setToLowercase(true)
      .setPattern("\\b[a-z0-9]+")
      .setInputCol("doc")
      .setOutputCol("tokens")
    docTokenizer
  }
}

class TokenAssembler(override val uid: String) extends Transformer {
  def this() = this(Identifiable.randomUID("tokenAssember"))

  final val inputCols: StringArrayParam = new StringArrayParam(this, "inputCols", "input column names")

  final def getInputCols: Array[String] = $(inputCols)

  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  setDefault(outputCol, uid + "__output")

  final def getOutputCol: String = $(outputCol)

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def biGramCombiner = (x: Seq[String], y: Seq[String]) => {
    x ++ y
  }

  def triGramCombiner = (x: Seq[String], y: Seq[String], z: Seq[String]) => {
    x ++ y ++ z
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputColName = $(outputCol)
    val combiner = $(inputCols).length match {
      case 2 => biGramCombiner
      case 3 => triGramCombiner
      case _ => throw new SparkException("The input columns should be of size 2 or 3")
    }
    val tranformUDF = udf(combiner, ArrayType(StringType))
    val newDs = $(inputCols).length match {
      case 2 =>
        dataset.withColumn(outputColName, tranformUDF(dataset("unigrams"), dataset("grams_2")))
      case 3 =>
        dataset.withColumn(outputColName, tranformUDF(dataset("unigrams"), dataset("grams_2"), dataset("grams_3")))
    }
    newDs
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    val outputColName = $(outputCol)
    StructType(schema.fields :+ StructField(outputColName, ArrayType(StringType, containsNull = false), nullable = true))
  }
}


case class NgramTokenizer(n: Int) extends DocGenerator {

  override def tokenizer: PipelineStage = {
    val docTokenizer = new RegexTokenizer()
      .setGaps(false)
      .setMinTokenLength(3)
      .setToLowercase(true)
      .setPattern("\\b[a-z0-9]+")
      .setInputCol("doc")
      .setOutputCol("unigrams")

    val grams: Seq[PipelineStage] = {
      for {
        i <- 2 to n
      } yield {
        val gramTokenizer = new NGram()
          .setInputCol("unigrams")
          .setOutputCol(s"grams_$i")
          .setN(i)
        gramTokenizer
      }
    }

    val gram_cols: Array[String] = (2 to n).map(x => s"grams_$x").toArray
    val assembler = new TokenAssembler()
      .setInputCols(Array("unigrams") ++ gram_cols)
      .setOutputCol("tokens")

    val tokenPipeline = new Pipeline()
    tokenPipeline.setStages(Array(docTokenizer) ++ grams ++ Array(assembler))
    tokenPipeline
  }
}

object BigramTokenizer extends NgramTokenizer(2)

object TrigramTokenizer extends NgramTokenizer(3)

