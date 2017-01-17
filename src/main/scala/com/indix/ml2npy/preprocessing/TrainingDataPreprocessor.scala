package com.indix.ml2npy.preprocessing

import com.indix.ml2npy.text.CooccurrenceTokenizer
import org.apache.spark.SparkException
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}

trait DocGenerator[T] {

  def main(args: Array[String]): Unit = {

    val inputPath: String = args(0)
    val outputPath: String = args(1)
    val numParts: Int = args(2).toInt

    val spark = SparkSession.builder().appName("Ml2Npy").getOrCreate()
    prepare(spark, inputPath, outputPath, numParts)
  }

  def tokenizer: PipelineStage

  def pipeline(tokenizer:PipelineStage):Pipeline

  def readRecords(spark: SparkSession, inputPath: String):Dataset[T]

  def writeRecords(spark: SparkSession, sampledRecords: Dataset[T], outputPath: String, pipeline: Pipeline, numParts: Int)

  def prepare(spark: SparkSession, inputPath: String, outputPath: String, numParts: Int): Unit = {
    val records = readRecords(spark,inputPath)
    val pp = pipeline(tokenizer)
    writeRecords(spark,records,outputPath,pp,numParts)
  }
}

trait UnigramTokens[T <: DocGenerator[T]] extends DocGenerator[T]{

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

trait CooccTokens[T <: DocGenerator[T]] extends DocGenerator[T] {

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

trait NgramTokenizer[T <: DocGenerator[T]] extends DocGenerator[T]{

  val n:Int

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

