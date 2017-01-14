package com.indix.ml2npy.preprocessing

import com.indix.ml2npy.hadoop.NpyOutPutFormat
import com.indix.ml2npy.text.CooccurrenceTokenizer
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.{Dataset, SparkSession}

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
    val trainingRecords =readRecords(inputPath)
    val toplevelTrainingRecords=sampleRecords(trainingRecords)
    writeRecords(toplevelTrainingRecords,outputPath)
  }

  def tokenizer: PipelineStage

  def pipeline: Pipeline

  def readRecords(inputPath:String):Dataset[TrainingRecord]

  def sampleRecords(trainingRecords:Dataset[TrainingRecord]):Dataset[SimpleTrainingRecord]

  def writeRecords(topLevelTrainingRecords: Dataset[SimpleTrainingRecord],outputPath:String)
}