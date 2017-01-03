package com.github.vumaasha.ml2npy

import com.github.vumaasha.ml2npy.hadoop.NpyOutPutFormat
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.SparkSession
import scala.util._


case class TrainingRecord(storeId: Long, url: String, title: String, breadCrumbs: String, brandText: String, categoryPath: String, price: Double, isbn: String, specificationText: String, leafId: String,
                          topLevelId: String)

case class SimpleTrainingRecord(url: String, doc: String, topLevelId: Short, categoryId: Short)

case class TFIdfModel(vocab: Array[String], idf: Vector)

/**
  * Created by vumaasha on 28/12/16.
  */
object ml2NpyTester {


  def main(args: Array[String]): Unit = {

    val inputPath: String = args(0)
    val outputPath: String = args(1)
    val numParts: Int = args(2).toInt

    val spark = SparkSession.builder().appName("Ml2Npy").getOrCreate()

    import spark.implicits._
    val trainingRecords = spark.read.json(inputPath).as[TrainingRecord].cache()

    val toplevelTrainingRecords = trainingRecords.mapPartitions(iterator => {
      val bcSelector = new Random(11)
      val brandSelector = new Random(17)
      val specSelector = new Random(23)
      for {
        record <- iterator
      } yield {
        val isBc = bcSelector.nextInt(3) == 0
        val isBrand = brandSelector.nextInt(2) == 0
        val isSpecs = bcSelector.nextInt(2) == 0
        val doc = {
          record.title + {
            if (isBrand) " " + record.brandText else " "
          } + {
            if (isBc) " " + record.breadCrumbs else " "
          } + {
            if (isSpecs) " " + record.specificationText else " "
          }
        }
        SimpleTrainingRecord(record.url, doc, Integer.parseInt(record.topLevelId).toShort, Integer.parseInt(record.leafId).toShort)
      }
    })

    val docTokenizer = new RegexTokenizer()
      .setGaps(false)
      .setMinTokenLength(3)
      .setToLowercase(true)
      .setPattern("\\b[a-z0-9]+")
      .setInputCol("doc")
      .setOutputCol("tokens")

    val docCV = new CountVectorizer()
      .setMinDF(100)
      .setInputCol("tokens")
      .setOutputCol("tokenCounts")

    val docIDF = new IDF()
      .setInputCol("tokenCounts")
      .setOutputCol("tokenIDF")

    val normalizer = new Normalizer()
      .setInputCol("tokenIDF")
      .setOutputCol("normalizedTokenIDF")

    val pipeline = new Pipeline()
      .setStages(Array(docTokenizer, docCV, docIDF, normalizer))

    val ppModel = pipeline.fit(toplevelTrainingRecords)

    val transformedTopData = ppModel.transform(toplevelTrainingRecords)

    transformedTopData.select("doc", "tokens", "normalizedTokenIDF").show()


    val idfData = transformedTopData.select("normalizedTokenIDF", "topLevelId", "categoryId")
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
    val tfidf: TFIdfModel = TFIdfModel(cvModel.vocabulary, idfModel.idf)
    val data: Seq[TFIdfModel] = Seq(tfidf)
    val tfidfDF = spark.createDataFrame[TFIdfModel](data).coalesce(1).toDF()
    tfidfDF.write.json(s"$outputPath/tfidfModel")


  }

}
